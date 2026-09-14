#! /usr/bin/env python

import subprocess
import os
import time
import redis
import middleware as mw


class DriverMicrophone:
    def __init__(self):
        self.node = mw.Node("driver_microphone")
        self.microphone = mw.Microphone()
        self.microphone_target = os.environ.get("MICROPHONE_TARGET")
        
        # Connect to Redis
        self.redis_client = redis.Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", 6379)),
            db=0
        )
        self.recording_process = None
        
        # 1280 samples * 2 bytes/sample (16-bit) = 2560 bytes chunk size
        self.chunk_bytes = 2 * 2560 

    def start_streaming_audio(self):
        """Spawns pw-record sending raw PCM bytes directly to stdout."""
        cmd = [
            "pw-record", 
            "--format=s16", 
            "--channels=1", 
            "--rate=16000", 
            "-"  # Send raw binary stream to stdout
        ]
        
        if self.microphone_target:
            cmd.insert(1, "--target")
            cmd.insert(2, self.microphone_target)

        self.recording_process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.DEVNULL, 
            bufsize=0
        )
        self.microphone.is_recording = True

    def _read_exact(self, num_bytes):
        """Helper to guarantee reading full byte chunks from stdout stream."""
        data = bytearray()
        while len(data) < num_bytes:
            packet = self.recording_process.stdout.read(num_bytes - len(data))
            if not packet:
                break
            data.extend(packet)
        return bytes(data)

    def stop_streaming_audio(self):
        if self.recording_process:
            self.recording_process.terminate()
            try:
                self.recording_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.recording_process.kill()
                self.recording_process.wait()
            self.recording_process = None
        self.microphone.is_recording = False

    def run(self):
        try:
            self.microphone.ready = True
            self.start_streaming_audio()
            
            while not self.node.is_shutdown():
                if self.recording_process and self.recording_process.stdout:
                    # Guarantee reading exactly self.chunk_bytes
                    raw_pcm_data = self._read_exact(self.chunk_bytes)
                    
                    if len(raw_pcm_data) == self.chunk_bytes:
                        # maxlen=100 keeps up to 100 entries in the stream history
                        self.redis_client.xadd(
                            "robot:mic_stream", 
                            {"pcm": raw_pcm_data}, 
                            maxlen=100, 
                            approximate=True
                        )
                else:
                    time.sleep(0.01)

        except KeyboardInterrupt:
            pass
        finally:
            self.stop_streaming_audio()
            self.node.shutdown()


if __name__ == "__main__":
    driver = DriverMicrophone()
    driver.run()