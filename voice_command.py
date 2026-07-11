import speech_recognition as sr
import pyttsx3

# ==========================================
# Text-to-Speech Engine
# ==========================================

engine = pyttsx3.init()

engine.setProperty("rate", 170)

voices = engine.getProperty("voices")

# Male voice
engine.setProperty("voice", voices[0].id)


# ==========================================
# Speak Function
# ==========================================

def speak(text):
    print("Assistant :", text)
    engine.say(text)
    engine.runAndWait()


# ==========================================
# Voice Recognition Function
# ==========================================

def listen_command():

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:

        print("\nListening...")

        recognizer.adjust_for_ambient_noise(source, duration=1)

        audio = recognizer.listen(source)

    try:

        command = recognizer.recognize_google(audio)

        command = command.lower()

        print("You Said :", command)

        # -------------------------
        # Valid Commands
        # -------------------------

        if "forward" in command:
            speak("Moving Forward")
            return "FORWARD"

        elif "backward" in command:
            speak("Moving Backward")
            return "BACKWARD"

        elif "left" in command:
            speak("Turning Left")
            return "LEFT"

        elif "right" in command:
            speak("Turning Right")
            return "RIGHT"

        elif "stop" in command:
            speak("Stopping")
            return "STOP"

        else:
            speak("Unknown Command")
            return None

    except sr.UnknownValueError:

        speak("Sorry, I didn't understand.")

    except sr.RequestError:

        speak("Internet connection required.")

    return None


# ==========================================
# Testing
# ==========================================

if __name__ == "__main__":

    print("=" * 50)
    print("VOICE COMMAND TEST")
    print("=" * 50)

    while True:

        command = listen_command()

        if command:

            print("Detected :", command)

            if command == "STOP":
                break