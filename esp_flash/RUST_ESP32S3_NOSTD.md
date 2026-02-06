# Rust for ESP32S3 (no_std)

This guide documents the setup and best practices for running Rust on ESP32S3 in a `no_std` environment, as used in `esp_flash`.

## 1. Environment Setup

To develop for ESP32S3 with Rust, you need the Espressif Rust toolchain.

### Install `espup`
```bash
cargo install espup
espup install
. ~/export-esp.sh
```

### linker Scripts
The memory layout is defined by `memory.x` (usually provided by `esp-hal` or generated).
Ensure that your `.cargo/config.toml` includes the target specific runner:

```toml
[target.xtensa-esp32s3-none-elf]
runner = "espflash flash --monitor"
```

## 2. Memory Management in `no_std`

### Stack
The default stack size for the main task in `no_std` is often limited (detectable by `_stack_start - _stack_end`).
Large stack allocations (like 36KB image buffers) will cause **Stack Overflow** and immediate reboot (Guru Meditation Error).

**Recommendation:**
- Avoid large variables on the stack.
- Use `static mut` or `lazy_static!` (if available) for large buffers.
- If using `alloc`, use the heap.
- **Note**: Rust 2024 denies `static_mut_refs` by default. Use `#[allow(static_mut_refs)]` carefully or usage `UnsafeCell` / `addr_of_mut!`.

### Heap (Optional)
To use `Box`, `Vec`, etc., you need to initialize a global allocator.
`esp-alloc` is commonly used.

```rust
use esp_alloc::EspHeap;

#[global_allocator]
static ALLOCATOR: EspHeap = EspHeap::empty();

fn init_heap() {
    const HEAP_SIZE: usize = 32 * 1024;
    static mut HEAP: [u8; HEAP_SIZE] = [0; HEAP_SIZE];
    unsafe { ALLOCATOR.init(HEAP.as_mut_ptr(), HEAP_SIZE) }
}
```

## 3. IRAM (Instruction RAM)
Code runs from Flash by default (slower). Critical code (interrupt handlers, tight inference loops) should run from IRAM.

To place a function in IRAM:
```rust
#[link_section = ".rwtext"]
fn fast_inference() {
    // ...
}
```
*Note: `esp-hal` attributes might vary by version. Check specific HAL docs.*

## 4. Optimization
In `Cargo.toml`:
```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
```

## 5. Microflow Specifics
- `Buffer2D` / `Buffer4D` are large arrays.
- Passing them by value copies memory.
- Prefer `static` allocation for inputs/outputs if they persist.
