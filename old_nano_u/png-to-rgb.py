#!/usr/bin/env python3
"""
png_to_rgb8.py

Minimal PNG -> raw RGB8 converter using only Python standard library.
Outputs raw bytes: R G B R G B ... (width * height * 3 bytes).

Usage:
    python png-to-rgb.py input.png output.rgb
"""

import sys
import struct
import zlib

PNG_SIG = b'\x89PNG\r\n\x1a\n'

# Paeth predictor
def paeth(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c

def unfilter_scanline(filter_type, scanline, prev_scanline, bpp):
    """Apply PNG filter to one scanline.
    scanline: bytes (without leading filter byte)
    prev_scanline: bytes or None
    bpp: bytes per pixel
    """
    out = bytearray(len(scanline))
    if filter_type == 0:  # None
        return bytearray(scanline)
    elif filter_type == 1:  # Sub
        for i in range(len(scanline)):
            left = out[i - bpp] if i >= bpp else 0
            out[i] = (scanline[i] + left) & 0xFF
        return out
    elif filter_type == 2:  # Up
        for i in range(len(scanline)):
            up = prev_scanline[i] if prev_scanline is not None else 0
            out[i] = (scanline[i] + up) & 0xFF
        return out
    elif filter_type == 3:  # Average
        for i in range(len(scanline)):
            left = out[i - bpp] if i >= bpp else 0
            up = prev_scanline[i] if prev_scanline is not None else 0
            out[i] = (scanline[i] + ((left + up) // 2)) & 0xFF
        return out
    elif filter_type == 4:  # Paeth
        for i in range(len(scanline)):
            left = out[i - bpp] if i >= bpp else 0
            up = prev_scanline[i] if prev_scanline is not None else 0
            up_left = prev_scanline[i - bpp] if (prev_scanline is not None and i >= bpp) else 0
            out[i] = (scanline[i] + paeth(left, up, up_left)) & 0xFF
        return out
    else:
        raise ValueError("Unknown filter type: %d" % filter_type)

def parse_png(data):
    # Validate signature
    if not data.startswith(PNG_SIG):
        raise ValueError("Not a PNG file (bad signature)")

    offset = len(PNG_SIG)
    ihdr = None
    idat_chunks = []
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("Truncated PNG")
        length = struct.unpack(">I", data[offset:offset+4])[0]
        ctype = data[offset+4:offset+8].decode('ascii')
        chunk_data = data[offset+8:offset+8+length]
        # crc = data[offset+8+length: offset+12+length]  # ignored here
        offset += 12 + length

        if ctype == 'IHDR':
            if length != 13:
                raise ValueError("IHDR has unexpected length")
            (width, height, bit_depth, color_type,
             compression, filter_method, interlace) = struct.unpack(">IIBBBBB", chunk_data)
            ihdr = dict(width=width, height=height, bit_depth=bit_depth,
                        color_type=color_type, compression=compression,
                        filter_method=filter_method, interlace=interlace)
        elif ctype == 'IDAT':
            idat_chunks.append(chunk_data)
        elif ctype == 'IEND':
            break
        else:
            # ignore other chunks
            continue

    if ihdr is None:
        raise ValueError("IHDR chunk missing")

    return ihdr, b"".join(idat_chunks)

def png_to_rgb_bytes(png_bytes):
    ihdr, idat_concat = parse_png(png_bytes)

    width = ihdr['width']
    height = ihdr['height']
    bit_depth = ihdr['bit_depth']
    color_type = ihdr['color_type']
    interlace = ihdr['interlace']

    if interlace != 0:
        raise NotImplementedError("Interlaced PNGs are not supported")

    if bit_depth != 8:
        raise NotImplementedError("Only 8-bit depth PNGs supported (bit_depth=%d)" % bit_depth)

    # color_type: 2 = truecolor RGB, 6 = truecolor with alpha (RGBA)
    if color_type == 2:
        channels = 3
    elif color_type == 6:
        channels = 4
    else:
        raise NotImplementedError("Only color types 2 (RGB) and 6 (RGBA) are supported. Got: %d" % color_type)

    bpp = channels  # bytes per pixel (since bit_depth is 8)
    bytes_per_scanline = width * channels

    # Decompress IDAT stream
    decompressed = zlib.decompress(idat_concat)

    # PNG scanlines: each scanline starts with a filter byte, then pixel bytes
    expected = (1 + bytes_per_scanline) * height
    if len(decompressed) != expected:
        # Some PNGs may include extra data (e.g. ancillary), but usually the size matches.
        # We'll do a best-effort: if larger, only read the required portion.
        if len(decompressed) < expected:
            raise ValueError("Decompressed IDAT size too small (got %d, need %d)" % (len(decompressed), expected))
        decompressed = decompressed[:expected]

    out = bytearray()
    prev = None
    pos = 0
    for row in range(height):
        filter_type = decompressed[pos]
        pos += 1
        scanline = decompressed[pos: pos + bytes_per_scanline]
        pos += bytes_per_scanline
        recon = unfilter_scanline(filter_type, scanline, prev, bpp)
        prev = recon
        if channels == 3:
            out.extend(recon)
        else:  # channels == 4 -> RGBA, drop alpha
            # recon is R G B A R G B A ...
            # keep R G B every 4 bytes
            out.extend(recon[i] for i in range(0, len(recon), 4))
    return bytes(out), width, height

def main():
    if len(sys.argv) != 3:
        print("Usage: python png_to_rgb8.py input.png output.rgb")
        sys.exit(2)

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    with open(in_path, 'rb') as f:
        png_bytes = f.read()

    try:
        rgb_bytes, w, h = png_to_rgb_bytes(png_bytes)
    except Exception as e:
        print("Error:", e)
        sys.exit(1)

    with open(out_path, 'wb') as f:
        f.write(rgb_bytes)

    print(f"Wrote raw RGB8 to {out_path} ({w}x{h}, {len(rgb_bytes)} bytes)")

if __name__ == "__main__":
    main()
