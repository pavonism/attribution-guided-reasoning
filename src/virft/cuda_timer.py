import torch
import os


class CUDATimer:
    def __init__(self):
        self.start_events = {}
        self.end_events = {}
        self.is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0

    def start(self, name):
        """Places a start marker in the GPU queue."""
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.start_events[name] = event

    def stop(self, name):
        """Places a stop marker in the GPU queue."""
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.end_events[name] = event

    def log_stats(self, step, log_freq=1):
        """Synchronizes the GPU and prints the average times."""
        if step % log_freq == 0 and self.is_main_process:
            # We MUST synchronize the GPU here to calculate the final times
            torch.cuda.synchronize()

            print(f"\n----------------- (Step {step}) -----------------")
            for name in self.start_events.keys():
                if name not in self.end_events:
                    print(f"⚠️ No stop event recorded for '{name}'")
                    continue

                # elapsed_time returns milliseconds, we convert to seconds
                ms = self.start_events[name].elapsed_time(self.end_events[name])
                print(f"➤ {name}: {ms / 1000:.4f} seconds")
            print("-------------------------------------------\n")
