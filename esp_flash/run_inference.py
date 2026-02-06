
import subprocess
import sys
import time
import os
import pty
import select

def main():
    cmd = ["espflash", "flash", "--monitor", "target/xtensa-esp32s3-none-elf/release/analysis"]
    log_file = "stack_log.txt"
    MaxRetries = 3
    
    for attempt in range(MaxRetries):
        print(f"Executing: {' '.join(cmd)} (Attempt {attempt+1}/{MaxRetries})")
        print(f"Logging to: {log_file}")
        
        # Create PTY to preserve coloring and output behavior
        master, slave = pty.openpty()
        
        # Start process
        p = subprocess.Popen(
            cmd,
            stdout=slave,
            stderr=subprocess.STDOUT,
            close_fds=True
        )
        os.close(slave)  # Close slave in parent
        
        # Output file (overwrite for each attempt to keep clean log)
        f_log = open(log_file, "wb")
        
        buffer = b""
        start_time = time.time()
        success = False
        port_error = False
        
        try:
            while True:
                # Check if process exited
                if p.poll() is not None:
                    break
                    
                # Read from master
                # Use a larger timeout for responsiveness
                r, _, _ = select.select([master], [], [], 0.1)
                if master in r:
                    try:
                        data = os.read(master, 1024)
                    except OSError:
                        break
                        
                    if not data:
                        break
                        
                    # Write to stdout (terminal)
                    os.write(sys.stdout.fileno(), data)
                    # Write to log file
                    f_log.write(data)
                    f_log.flush()
                    
                    buffer += data
                    
                    # Check for completion marker
                    if b"ANALYSIS_DONE" in buffer:
                        print("\n\nFound completion marker! Stopping process...")
                        success = True
                        break
                    
                    # Check for port errors to trigger retry
                    if b"Failed to open serial port" in buffer or b"Error while connecting" in buffer:
                        port_error = True
                        # Don't break immediately, let process exit or we kill it
                        
                    # Check for App Descriptor Missing error (fatal)
                    if b"ESP-IDF App Descriptor" in buffer and b"missing" in buffer:
                         # This is a build/linker error on the device side (or flashing side check)
                         # No point retrying indefinitely if binary is bad
                         # But wait, flashing might fail?
                         pass # Let it finish
            
            # Timeout safety (5 minutes)
            if time.time() - start_time > 300:
                print("\nTimeout reached!")
                break
                
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            sys.exit(130)
            
        finally:
            f_log.close()
            # Terminate process
            if p.poll() is None:
                p.terminate()
                time.sleep(0.5)
                if p.poll() is None:
                    p.kill()
            
            # Cleanup PTY
            os.close(master)
        
        # Outcome processing (Outside finally)
        if success:
            sys.exit(0)
        
        if port_error and attempt < MaxRetries - 1:
            print("\n⚠️  Port error detected. Retrying in 2 seconds...")
            time.sleep(2)
            continue
        
    print("Did not complete successfully after retries.")
    sys.exit(1)

if __name__ == "__main__":
    main()
