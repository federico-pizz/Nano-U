//! OV2640 live-capture driver for the Goouuu ESP32-S3-CAM (`no_std`, esp-hal).
//!
//! This is the only *new hardware* path in the project. It captures a live
//! QQVGA (160×120) RGB565 frame over the ESP32-S3 `LCD_CAM` (DVP) peripheral via
//! DMA, then box-downscales 2×2 and INT8-quantizes it straight into the existing
//! `IMG_H`×`IMG_W` (60×80) input buffer that [`crate::preprocess_rgb`] would
//! otherwise fill from a baked image. The model and its activations are
//! untouched — only the *frame source* changes.
//!
//! ## Memory
//! The 38 400-byte RGB565 frame lives in **PSRAM** (allocated once at startup via
//! the global PSRAM allocator set up in `bin/online.rs`), not internal SRAM, so
//! the ~30–40 KB SRAM headroom and the model activations are left alone. The
//! frame is written once by DMA (background) and read once during downscale
//! (~1 ms), so PSRAM costs a sliver of latency but **does not reduce throughput**
//! — the loop stays inference-bound (~800 ms/frame).
//!
//! ## Hardware-only / validation notes
//! This module is **not exercised in CI** (it needs the esp toolchain and a
//! physical camera). Three things must be confirmed on-device when first
//! bringing it up; each is flagged inline below:
//!   1. **Register table** — [`OV2640_RGB565_QQVGA`] is ported from Espressif's
//!      `esp32-camera` `ov2640_settings.h`. If capture is blank/garbled, diff it
//!      against upstream for your sensor revision.
//!   2. **RGB565 byte order** — DMA may deliver the two bytes hi/lo or lo/hi.
//!      Flip [`SWAP_RGB565_BYTES`] if colours look wrong.
//!   3. **PSRAM↔DMA cache coherency** — the DMA buffer must be cache-invalidated
//!      before the CPU reads it; esp-hal's `DmaRxBuf` is expected to handle this
//!      for external memory, but verify the frame isn't stale.

use esp_hal::{
    delay::Delay,
    dma::{DmaBufError, DmaError, DmaRxBuf},
    dma_descriptors,
    i2c::master::{Config as I2cConfig, I2c},
    lcd_cam::{
        cam::{Camera, Config as CamConfig},
        LcdCam,
    },
    time::Rate,
    Blocking,
};

use microflow::buffer::Buffer2D;

use crate::{IMG_H, IMG_W};

/// Per-channel [0..255] → i8 LUTs produced by [`crate::build_quant_luts`].
pub type QuantLuts = ([i8; 256], [i8; 256], [i8; 256]);

// ── Capture geometry ─────────────────────────────────────────────────────────
// OV2640's smallest reliable RGB565 frame is QQVGA (160×120); direct 80×60
// windowing trips NO-EOI/FB-OVF on the sensor DSP. QQVGA is exactly 2× the model
// input in each axis, so a clean 2×2 box filter lands on 60×80.
pub const CAM_W: usize = IMG_W * 2; // 160
pub const CAM_H: usize = IMG_H * 2; // 120
/// RGB565 = 2 bytes/pixel.
pub const FRAME_BYTES: usize = CAM_W * CAM_H * 2;

/// OV2640 SCCB (I²C) 7-bit address.
const OV2640_ADDR: u8 = 0x30;
/// Bank-select register: `0x00` = DSP bank, `0x01` = sensor bank.
const REG_BANK_SEL: u8 = 0xFF;

/// Flip if captured colours are wrong (see module note #2).
const SWAP_RGB565_BYTES: bool = false;

/// SCCB master-clock fed to the OV2640 XCLK. 20 MHz is the standard DVP rate.
const XCLK_HZ: u32 = 20_000_000;

// ── OV2640 register init (faithful port of Espressif esp32-camera) ───────────
//
// Entries are `[reg, val]`. `[REG_BANK_SEL, 0x00|0x01]` switches banks
// (0x00 = DSP, 0x01 = sensor). This reproduces, *in order*, the exact register
// writes the esp32-camera runtime emits for **RGB565 @ QQVGA (160×120)**:
//
//   reset()            → COM7 soft-reset (done in `new()` before this table)
//                        + `ov2640_settings_cif`  (the bulk below, ending with
//                          the DSP-enable tail: CTRL1=0xfd, RESET=0x00, R_BYPASS=0x00)
//   set_pixformat()    → `ov2640_settings_rgb565`
//   set_framesize()    → set_window(): R_BYPASS bypass, `ov2640_settings_to_cif`,
//                        computed window/zoom regs, CLKRC + R_DVP_SP clock, R_BYPASS enable
//   set_pixformat()    → `ov2640_settings_rgb565` again  (its final RESET=0x00
//                        releases the DVP — this is what actually starts streaming)
//
// The two halves are split by a 10 ms settle (see [`OnboardCamera::new`]) exactly
// as the C driver does between set_window and its trailing set_pixformat.
//
// Window math for QQVGA (4:3 → full window 1600×1200, offset 0,0):
//   set_framesize: /4 → max 400×300 (clamped 296), offset 0,0
//   set_window:    /4 → HSIZE=0x64 VSIZE=0x4a; w,h /4 → ZMOW=0x28 ZMOH=0x1e
// Clock (CIF, non-JPEG, ESP32-S3): clk_2x=1 clk_div=3 → CLKRC=0x83;
//   pclk_auto=1 pclk_div=8 → R_DVP_SP=0x88.
// Source: espressif/esp32-camera `ov2640_settings.h` + `ov2640.c`. See note #1.
#[rustfmt::skip]
const OV2640_RGB565_QQVGA: &[[u8; 2]] = &[
    // ── reset(): ov2640_settings_cif ─────────────────────────────────────────
    [REG_BANK_SEL, 0x00],   // BANK_DSP
    [0x2c, 0xff],
    [0x2e, 0xdf],
    [REG_BANK_SEL, 0x01],   // BANK_SENSOR
    [0x3c, 0x32],
    [0x11, 0x01],           // CLKRC
    [0x09, 0x02],           // COM2: output drive x3
    [0x04, 0x28],           // REG04
    [0x13, 0xe5],           // COM8: default | BNDF | AGC | AEC
    [0x14, 0x48],           // COM9: AGC ceiling 8x
    [0x2c, 0x0c],
    [0x33, 0x78],
    [0x3a, 0x33],
    [0x3b, 0xfb],
    [0x3e, 0x00],
    [0x43, 0x11],
    [0x16, 0x10],
    [0x39, 0x92],
    [0x35, 0xda],
    [0x22, 0x1a],
    [0x37, 0xc3],
    [0x23, 0x00],
    [0x34, 0xc0],           // ARCOM2
    [0x06, 0x88],
    [0x07, 0xc0],
    [0x0d, 0x87],           // COM4
    [0x0e, 0x41],
    [0x4c, 0x00],
    [0x4a, 0x81],
    [0x21, 0x99],
    [0x24, 0x40],           // AEW
    [0x25, 0x38],           // AEB
    [0x26, 0x82],           // VV
    [0x5c, 0x00],
    [0x63, 0x00],
    [0x61, 0x70],           // HISTO_LOW
    [0x62, 0x80],           // HISTO_HIGH
    [0x7c, 0x05],
    [0x20, 0x80],
    [0x28, 0x30],
    [0x6c, 0x00],
    [0x6d, 0x80],
    [0x6e, 0x00],
    [0x70, 0x02],
    [0x71, 0x94],
    [0x73, 0xc1],
    [0x3d, 0x34],
    [0x5a, 0x57],
    [0x4f, 0xbb],           // BD50
    [0x50, 0x9c],           // BD60
    [0x12, 0x20],           // COM7: CIF
    [0x17, 0x11],           // HSTART
    [0x18, 0x43],           // HSTOP
    [0x19, 0x00],           // VSTART
    [0x1a, 0x25],           // VSTOP
    [0x32, 0x89],           // REG32: CIF
    [0x37, 0xc0],
    [0x4f, 0xca],           // BD50
    [0x50, 0xa8],           // BD60
    [0x6d, 0x00],
    [0x3d, 0x38],
    [REG_BANK_SEL, 0x00],   // BANK_DSP
    [0xe5, 0x7f],
    [0xf9, 0xc0],           // MC_BIST: reset | boot_rom_sel
    [0x41, 0x24],
    [0xe0, 0x14],           // RESET: JPEG | DVP
    [0x76, 0xff],
    [0x33, 0xa0],
    [0x42, 0x20],
    [0x43, 0x18],
    [0x4c, 0x00],
    [0x87, 0x50],           // CTRL3: WPC_EN | 0x10
    [0x88, 0x3f],
    [0xd7, 0x03],
    [0xd9, 0x10],
    [0xd3, 0x82],           // R_DVP_SP: auto | 0x02
    [0xc8, 0x08],
    [0xc9, 0x80],
    [0x7c, 0x00],           // BPADDR
    [0x7d, 0x00],           // BPDATA
    [0x7c, 0x03],
    [0x7d, 0x48],
    [0x7d, 0x48],
    [0x7c, 0x08],
    [0x7d, 0x20],
    [0x7d, 0x10],
    [0x7d, 0x0e],
    [0x90, 0x00],
    [0x91, 0x0e],
    [0x91, 0x1a],
    [0x91, 0x31],
    [0x91, 0x5a],
    [0x91, 0x69],
    [0x91, 0x75],
    [0x91, 0x7e],
    [0x91, 0x88],
    [0x91, 0x8f],
    [0x91, 0x96],
    [0x91, 0xa3],
    [0x91, 0xaf],
    [0x91, 0xc4],
    [0x91, 0xd7],
    [0x91, 0xe8],
    [0x91, 0x20],
    [0x92, 0x00],
    [0x93, 0x06],
    [0x93, 0xe3],
    [0x93, 0x05],
    [0x93, 0x05],
    [0x93, 0x00],
    [0x93, 0x04],
    [0x93, 0x00],
    [0x93, 0x00],
    [0x93, 0x00],
    [0x93, 0x00],
    [0x93, 0x00],
    [0x93, 0x00],
    [0x93, 0x00],
    [0x96, 0x00],
    [0x97, 0x08],
    [0x97, 0x19],
    [0x97, 0x02],
    [0x97, 0x0c],
    [0x97, 0x24],
    [0x97, 0x30],
    [0x97, 0x28],
    [0x97, 0x26],
    [0x97, 0x02],
    [0x97, 0x98],
    [0x97, 0x80],
    [0x97, 0x00],
    [0x97, 0x00],
    [0xa4, 0x00],
    [0xa8, 0x00],
    [0xc5, 0x11],
    [0xc6, 0x51],
    [0xbf, 0x80],
    [0xc7, 0x10],           // AWB
    [0xb6, 0x66],
    [0xb8, 0xa5],
    [0xb7, 0x64],
    [0xb9, 0x7c],
    [0xb3, 0xaf],
    [0xb4, 0x97],
    [0xb5, 0xff],
    [0xb0, 0xc5],
    [0xb1, 0x94],
    [0xb2, 0x0f],
    [0xc4, 0x5c],
    [0xc3, 0xfd],           // CTRL1: enable DSP modules
    [0x7f, 0x00],
    [0xe5, 0x1f],
    [0xe1, 0x67],
    [0xdd, 0x7f],
    [0xda, 0x00],           // IMAGE_MODE
    [0xe0, 0x00],           // RESET: release
    [0x05, 0x00],           // R_BYPASS: DSP enable

    // ── set_pixformat(RGB565): ov2640_settings_rgb565 ────────────────────────
    [REG_BANK_SEL, 0x00],   // BANK_DSP
    [0xe0, 0x04],           // RESET: DVP
    [0xda, 0x08],           // IMAGE_MODE: RGB565
    [0xd7, 0x03],
    [0xe1, 0x77],
    [0xe0, 0x00],           // RESET: release

    // ── set_framesize(QQVGA): set_window() ───────────────────────────────────
    [REG_BANK_SEL, 0x00],   // BANK_DSP
    [0x05, 0x01],           // R_BYPASS: DSP bypass (during reconfig)
    // ov2640_settings_to_cif
    [REG_BANK_SEL, 0x01],   // BANK_SENSOR
    [0x12, 0x20],           // COM7: CIF
    [0x03, 0x0a],           // COM1
    [0x32, 0x89],           // REG32: CIF
    [0x17, 0x11],           // HSTART
    [0x18, 0x43],           // HSTOP
    [0x19, 0x00],           // VSTART
    [0x1a, 0x25],           // VSTOP
    [0x4f, 0xca],           // BD50
    [0x50, 0xa8],           // BD60
    [0x5a, 0x23],
    [0x6d, 0x00],
    [0x3d, 0x38],
    [0x39, 0x92],
    [0x35, 0xda],
    [0x22, 0x1a],
    [0x37, 0xc3],
    [0x23, 0x00],
    [0x34, 0xc0],           // ARCOM2
    [0x06, 0x88],
    [0x07, 0xc0],
    [0x0d, 0x87],           // COM4
    [0x0e, 0x41],
    [0x4c, 0x00],
    [REG_BANK_SEL, 0x00],   // BANK_DSP
    [0xe0, 0x04],           // RESET: DVP (held during window setup)
    [0xc0, 0x32],           // HSIZE8
    [0xc1, 0x25],           // VSIZE8
    [0x8c, 0x00],           // SIZEL
    [0x51, 0x64],           // HSIZE
    [0x52, 0x4a],           // VSIZE
    [0x53, 0x00],           // XOFFL
    [0x54, 0x00],           // YOFFL
    [0x55, 0x00],           // VHYX
    [0x57, 0x00],           // TEST
    [0x86, 0x3d],           // CTRL2: DCW_EN | 0x1d
    [0x50, 0x80],           // CTRLI: LP_DP
    // computed win_regs (QQVGA)
    [REG_BANK_SEL, 0x00],   // BANK_DSP
    [0x51, 0x64],           // HSIZE  = max_x/4 = 0x64
    [0x52, 0x4a],           // VSIZE  = max_y/4 = 0x4a
    [0x53, 0x00],           // XOFFL
    [0x54, 0x00],           // YOFFL
    [0x55, 0x00],           // VHYX
    [0x57, 0x00],           // TEST
    [0x5a, 0x28],           // ZMOW = OUTW/4 = 0x28
    [0x5b, 0x1e],           // ZMOH = OUTH/4 = 0x1e
    [0x5c, 0x00],           // ZMHH
    // clock
    [REG_BANK_SEL, 0x01],   // BANK_SENSOR
    [0x11, 0x83],           // CLKRC: clk_2x | div 3
    [REG_BANK_SEL, 0x00],   // BANK_DSP
    [0xd3, 0x88],           // R_DVP_SP: pclk_auto | div 8
    [0x05, 0x00],           // R_BYPASS: DSP enable
];

// Trailing `set_pixformat(RGB565)` from set_window — applied after a 10 ms settle.
// Its final RESET=0x00 releases the DVP and starts the pixel stream.
#[rustfmt::skip]
const OV2640_RGB565_FINALIZE: &[[u8; 2]] = &[
    [REG_BANK_SEL, 0x00],   // BANK_DSP
    [0xe0, 0x04],           // RESET: DVP
    [0xda, 0x08],           // IMAGE_MODE: RGB565
    [0xd7, 0x03],
    [0xe1, 0x77],
    [0xe0, 0x00],           // RESET: release → DVP starts streaming
];

/// Errors surfaced while bringing the camera up or capturing a frame.
#[derive(Debug)]
pub enum CamError {
    /// SCCB probe read back an unexpected chip id (sensor not responding).
    BadChipId(u8, u8),
    /// An I²C/SCCB transaction failed.
    Sccb,
    /// Camera peripheral configuration failed.
    Config,
    /// DMA buffer construction failed (alignment / memory region).
    DmaBuf(DmaBufError),
    /// DMA transfer failed.
    Dma(DmaError),
}

/// Live OV2640 camera for the Goouuu ESP32-S3-CAM.
///
/// Holds the persistent LCD_CAM receiver (built once in [`build_camera`] and kept
/// alive so XCLK never stops) and its PSRAM-backed DMA receive buffer. Both live
/// in an `Option` so they can be moved into a transfer and back out on `stop()`.
pub struct OnboardCamera {
    camera: Option<Camera<'static>>,
    dma_buf: Option<DmaRxBuf>,
}

/// Build a fully-configured LCD_CAM DVP receiver for the Goouuu ESP32-S3-CAM,
/// stealing the peripheral + pin singletons.
///
/// Pins (verified ESP32-S3-EYE map, which the Goouuu board follows):
/// XCLK=15, PCLK=13, VSYNC=6, HREF=7, D0..D7 = 11,9,8,10,12,18,17,16.
///
/// Called once from [`OnboardCamera::new`]; the receiver is then kept alive for
/// the whole program and reused across `receive`/`stop` cycles. `with_master_clock`
/// enables the XCLK output permanently (not gated by `cam_start`), so the sensor
/// stays clocked and streaming between captures. `steal()` is sound here because
/// exactly one `Camera` (and thus one set of these singletons) is ever live.
fn build_camera() -> Result<Camera<'static>, CamError> {
    use esp_hal::peripherals;
    let lcd_cam = LcdCam::new(unsafe { peripherals::LCD_CAM::steal() });
    let camera = Camera::new(
        lcd_cam.cam,
        unsafe { peripherals::DMA_CH0::steal() },
        CamConfig::default().with_frequency(Rate::from_hz(XCLK_HZ)),
    )
    .map_err(|_| CamError::Config)?
    .with_master_clock(unsafe { peripherals::GPIO15::steal() })
    .with_pixel_clock(unsafe { peripherals::GPIO13::steal() })
    .with_vsync(unsafe { peripherals::GPIO6::steal() })
    .with_h_enable(unsafe { peripherals::GPIO7::steal() })
    .with_data0(unsafe { peripherals::GPIO11::steal() })
    .with_data1(unsafe { peripherals::GPIO9::steal() })
    .with_data2(unsafe { peripherals::GPIO8::steal() })
    .with_data3(unsafe { peripherals::GPIO10::steal() })
    .with_data4(unsafe { peripherals::GPIO12::steal() })
    .with_data5(unsafe { peripherals::GPIO18::steal() })
    .with_data6(unsafe { peripherals::GPIO17::steal() })
    .with_data7(unsafe { peripherals::GPIO16::steal() });
    Ok(camera)
}

impl OnboardCamera {
    /// Bring up XCLK, push the OV2640 RGB565/QQVGA register table over SCCB, and
    /// prepare the persistent PSRAM DMA buffer.
    ///
    /// `framebuffer` must be a `'static`, DMA-capable, cache-line (64-byte)
    /// aligned slice of **exactly [`FRAME_BYTES`]** bytes — the caller carves it
    /// out of the PSRAM region via `esp_hal::psram::psram_raw_parts`. No heap is
    /// used, so the rest of the firmware stays allocation-free.
    pub fn new(framebuffer: &'static mut [u8]) -> Result<Self, CamError> {
        let delay = Delay::new();

        // Build the receiver once to start XCLK: the OV2640 has no internal
        // oscillator and won't ACK on SCCB until its XCLK (CAM_CLK) is running.
        let camera = build_camera()?;
        delay.delay_millis(10);

        // ── SCCB: configure the sensor over I²C (SDA=4, SCL=5) ───────────────
        let mut sccb = I2c::new(
            unsafe { esp_hal::peripherals::I2C0::steal() },
            I2cConfig::default(),
        )
        .map_err(|_| CamError::Sccb)?
        .with_sda(unsafe { esp_hal::peripherals::GPIO4::steal() })
        .with_scl(unsafe { esp_hal::peripherals::GPIO5::steal() });

        // Software reset (PWDN/RESET aren't wired on this board): sensor bank,
        // COM7 |= 0x80, then let it settle.
        write_reg(&mut sccb, REG_BANK_SEL, 0x01)?;
        write_reg(&mut sccb, 0x12, 0x80)?;
        delay.delay_millis(10);

        // Probe the chip id (bank 1: PIDH=0x0A, PIDL=0x0B). OV2640 = 0x26 / 0x42.
        write_reg(&mut sccb, REG_BANK_SEL, 0x01)?;
        let pidh = read_reg(&mut sccb, 0x0A)?;
        let pidl = read_reg(&mut sccb, 0x0B)?;
        if pidh != 0x26 {
            return Err(CamError::BadChipId(pidh, pidl));
        }

        // Push the cif + rgb565 + window stages.
        for &[reg, val] in OV2640_RGB565_QQVGA {
            write_reg(&mut sccb, reg, val)?;
            delay.delay_micros(50);
        }
        // set_window ends by re-enabling the DSP; the C driver waits 10 ms before
        // its trailing set_pixformat, which releases the DVP from reset.
        delay.delay_millis(10);
        for &[reg, val] in OV2640_RGB565_FINALIZE {
            write_reg(&mut sccb, reg, val)?;
            delay.delay_micros(50);
        }
        // Let AEC/AGC converge before the first capture.
        delay.delay_millis(300);

        // Keep the camera alive for the whole program: XCLK stays running (the
        // sensor never loses its clock) and the LCD_CAM/DMA/pins stay configured.
        // Only the SCCB handle is dropped (GPIO4/5 aren't reused).
        drop(sccb);

        // ── Persistent DMA receive buffer ───────────────────────────────────
        assert_eq!(framebuffer.len(), FRAME_BYTES, "framebuffer must be FRAME_BYTES");
        let (rx_descriptors, _tx_descriptors) = dma_descriptors!(FRAME_BYTES, 0);
        let dma_buf = DmaRxBuf::new(rx_descriptors, framebuffer).map_err(CamError::DmaBuf)?;

        Ok(Self {
            camera: Some(camera),
            dma_buf: Some(dma_buf),
        })
    }

    /// Capture one live frame and write it, downscaled + INT8-quantized, into
    /// `out`. Reuses `out` and the DMA buffer in place — no per-frame allocation.
    ///
    /// The persistent `Camera` is moved into a DMA transfer and back out: start,
    /// poll `is_done()` until the buffer fills with one full frame, `stop()`, then
    /// read the (now complete, VSYNC-aligned) buffer. The camera is kept alive.
    pub fn capture_into(
        &mut self,
        luts: &QuantLuts,
        out: &mut Buffer2D<[i8; 3], IMG_H, IMG_W>,
    ) -> Result<(), CamError> {
        let dma_buf = self.dma_buf.take().expect("dma buf moved out");
        let camera = self.camera.take().expect("camera moved out");
        let delay = Delay::new();

        let transfer = match camera.receive(dma_buf) {
            Ok(t) => t,
            Err((e, camera, dma_buf)) => {
                self.camera = Some(camera);
                self.dma_buf = Some(dma_buf);
                return Err(CamError::Dma(e));
            }
        };

        // Wait for the frame to land: `cam_stop_en` clears `cam_start` (i.e.
        // `is_done()`) once the DMA buffer fills with one full frame. We start
        // mid-frame, so completion takes up to ~2 frame times; poll in small steps
        // up to a cap rather than blocking on a fixed delay. The loop is
        // inference-bound (~830 ms/frame) so this wait never gates throughput.
        const CAPTURE_POLL_STEP_MS: u32 = 5;
        const CAPTURE_TIMEOUT_MS: u32 = 250;
        let mut waited = 0;
        while !transfer.is_done() && waited < CAPTURE_TIMEOUT_MS {
            delay.delay_millis(CAPTURE_POLL_STEP_MS);
            waited += CAPTURE_POLL_STEP_MS;
        }

        let (camera, dma_buf) = transfer.stop();
        self.camera = Some(camera); // keep alive for the next capture

        // Invalidate the data cache for the framebuffer *after* the DMA write so
        // the CPU reads the freshly-captured PSRAM data, not a stale cached copy.
        // (ESP32-S3 ROM routine, already linked by esp-hal.)
        unsafe {
            let s = dma_buf.as_slice();
            Cache_Invalidate_Addr(s.as_ptr() as u32, s.len() as u32);
        }

        // Downscale + quantize straight from the captured frame into `out`.
        downscale_quantize(dma_buf.as_slice(), luts, out);

        self.dma_buf = Some(dma_buf);
        Ok(())
    }
}

// ESP32-S3 ROM data-cache invalidate by address range. Linked from ROM (also
// used internally by esp-hal). Used to drop stale cached lines for the PSRAM
// framebuffer after a DMA capture, before the CPU reads it.
unsafe extern "C" {
    fn Cache_Invalidate_Addr(addr: u32, size: u32);
}

/// Write one SCCB register (`[reg, val]`).
fn write_reg(sccb: &mut I2c<'_, Blocking>, reg: u8, val: u8) -> Result<(), CamError> {
    sccb.write(OV2640_ADDR, &[reg, val]).map_err(|_| CamError::Sccb)
}

/// Read one SCCB register (write address, then read one byte).
fn read_reg(sccb: &mut I2c<'_, Blocking>, reg: u8) -> Result<u8, CamError> {
    let mut buf = [0u8; 1];
    sccb.write_read(OV2640_ADDR, &[reg], &mut buf).map_err(|_| CamError::Sccb)?;
    Ok(buf[0])
}

/// Expand a 5-bit channel (0..31) to 8 bits.
#[inline]
fn expand5(v: u16) -> u8 {
    let v = v & 0x1f;
    ((v << 3) | (v >> 2)) as u8
}

/// Expand a 6-bit channel (0..63) to 8 bits.
#[inline]
fn expand6(v: u16) -> u8 {
    let v = v & 0x3f;
    ((v << 2) | (v >> 4)) as u8
}

/// 2×2 box-downscale a [`CAM_H`]×[`CAM_W`] RGB565 frame to [`IMG_H`]×[`IMG_W`]
/// and INT8-quantize through `luts`, writing in place into `out`.
///
/// Averaging is done in 8-bit channel space (expand RGB565 → RGB888, mean of the
/// 2×2 block, then LUT) so the result matches the host `preprocess_rgb` math up
/// to the box filter. `out[(h, w)] = [lut_r[r], lut_g[g], lut_b[b]]`.
fn downscale_quantize(
    frame: &[u8],
    luts: &QuantLuts,
    out: &mut Buffer2D<[i8; 3], IMG_H, IMG_W>,
) {
    let (lut_r, lut_g, lut_b) = luts;
    let row_bytes = CAM_W * 2;

    for h in 0..IMG_H {
        for w in 0..IMG_W {
            let mut r_sum: u16 = 0;
            let mut g_sum: u16 = 0;
            let mut b_sum: u16 = 0;

            // Accumulate the 2×2 source block.
            for dy in 0..2 {
                let row = (h * 2 + dy) * row_bytes;
                for dx in 0..2 {
                    let idx = row + (w * 2 + dx) * 2;
                    if idx + 1 >= frame.len() {
                        continue;
                    }
                    let (b0, b1) = if SWAP_RGB565_BYTES {
                        (frame[idx + 1], frame[idx])
                    } else {
                        (frame[idx], frame[idx + 1])
                    };
                    let px = ((b0 as u16) << 8) | b1 as u16;
                    r_sum += expand5(px >> 11) as u16;
                    g_sum += expand6(px >> 5) as u16;
                    b_sum += expand5(px) as u16;
                }
            }

            // Mean of the 4 samples, then INT8 quantize via LUT.
            let r = (r_sum / 4) as usize;
            let g = (g_sum / 4) as usize;
            let b = (b_sum / 4) as usize;
            out[(h, w)] = [lut_r[r], lut_g[g], lut_b[b]];
        }
    }
}
