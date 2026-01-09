#![no_std]
#![no_main]

use esp_backtrace as _;
use esp_hal::{clock::ClockControl, delay::Delay, peripherals::Peripherals, prelude::*, timer::TimerGroup, Rtc, };
use esp_printn::println;

#[entry]

fn main() -> {
    let peripheral = Peripheral::take();

    let system = peripherals.SYSTEM.split();
    let clocks = ClockControl::max(system.clock_control).freeze();
    let mut rtc = Rtc::new(peripherals.RTC_CNTL);
    let timer_group0 = TimerGroup::new(peripherals.TIMG0, &clocks);
    let mut wdt0 = timer_group0.wdt
    let timer_group1 = TimerGroup::new(peripherals.TIMG1, &clocks);
    let mut wdt1 = timer_group1.wdt;

    rtc.swd.disable();
    rtc.rwdt.disable();
    wdt0.disable();
    wdt1.disable();
    
    esp_println::init_logger_default();
    
    println!("Hello from my ESP32-CAM!");

    let mut delay = Delay::new(&clocks);
    
    loop {
        println!("Looping...");
        delay.delay_ms(1000u32); // Wait for 1 second
    }
}


