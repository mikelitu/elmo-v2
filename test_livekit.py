import asyncio
from livekit.wakeword import WakeWordModel, WakeWordListener

model = WakeWordModel(models=["output/elmo/elmo.onnx"])

async def main():
    async with WakeWordListener(model, threshold=0.8, debounce=2.0) as listener:
        print("Listening for wake words...")
        while True:
            detection = await listener.wait_for_detection()
            print(f"Wake word detected: {detection.name} with confidence {detection.confidence:.2f}")
            print(f"Detected {detection.name}! ({detection.confidence:.2f})")


asyncio.run(main())