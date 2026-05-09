#![no_std]

pub const fn parse_u32_const(s: &str) -> u32 {
    let bytes = s.as_bytes();
    let mut val: u32 = 0;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] >= b'0' && bytes[i] <= b'9' {
            val = val * 10 + (bytes[i] - b'0') as u32;
        }
        i += 1;
    }
    val
}

pub const fn str_to_i32_const(s: &str) -> i32 {
    let bytes = s.as_bytes();
    let mut val: i64 = 0;
    let mut neg = false;
    let mut i = 0;

    while i < bytes.len() && bytes[i] == b' ' {
        i += 1;
    }

    if i < bytes.len() && bytes[i] == b'-' {
        neg = true;
        i += 1;
    } else if i < bytes.len() && bytes[i] == b'+' {
        i += 1;
    }

    while i < bytes.len() {
        let b = bytes[i];
        if b >= b'0' && b <= b'9' {
            val = val * 10 + (b - b'0') as i64;
        } else if b == b'.' {
            break;
        }
        i += 1;
    }
    let res = if neg { -val } else { val };
    res as i32
}
