"""
List all available input microphones through PyAudio.

Usage:
    python list_microphones.py
"""

try:
    import pyaudio
except ImportError:
    print("PyAudio is not installed. Run: pip install pyaudio")
    raise SystemExit(1)


def main() -> None:
    pa = pyaudio.PyAudio()
    print("Available input microphones:")
    found = False

    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                found = True
                print(
                    f"[{i}] {info.get('name', 'Unknown')} | "
                    f"inputs={info.get('maxInputChannels')} | "
                    f"default_rate={int(info.get('defaultSampleRate', 0))}"
                )

        if not found:
            print("No input microphones found.")
    finally:
        pa.terminate()


if __name__ == "__main__":
    main()
