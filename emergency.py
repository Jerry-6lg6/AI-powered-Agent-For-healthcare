"""
emergency.py
------------
EmergencyAlert: handles all emergency detection and alerting.
- Plays a continuous alarm sound until staff press Enter to stop
- Speaks the patient name and location via TTS
- Prints a clear alert to the terminal
- Saves the event to the database
"""

import threading
import time
import numpy as np
import pygame


class EmergencyAlert:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._alarm_thread = None
        self._stop_alarm = threading.Event()

        # Initialise pygame mixer if not already done
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=sample_rate, size=-16,
                              channels=2, buffer=1024)

    def _generate_alarm_sound(self) -> pygame.mixer.Sound:
        """
        Generate a two-tone urgent alarm (alternating 880Hz / 1100Hz).
        Resembles a hospital call alarm.
        """
        tone_duration = 0.3           # seconds per tone
        n_samples = int(self.sample_rate * tone_duration)
        freqs = [880, 1100]           # Hz — alternating high tones
        volume = 0.8

        full_wave = []
        for freq in freqs:
            t = np.linspace(0, tone_duration, n_samples, endpoint=False)
            wave = (volume * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            full_wave.append(wave)

        combined = np.concatenate(full_wave)
        stereo = np.column_stack([combined, combined])
        return pygame.sndarray.make_sound(stereo)

    def _alarm_loop(self):
        """Play alarm continuously until _stop_alarm is set."""
        sound = self._generate_alarm_sound()
        while not self._stop_alarm.is_set():
            sound.play()
            # Wait for sound duration before replaying (0.6s = 2 x 0.3s tones)
            time.sleep(0.65)

    def _wait_for_staff(self, patient_name: str):
        """Block until staff press Enter to acknowledge the emergency."""
        print("\n" + "=" * 60)
        print("  *** EMERGENCY ALERT ***")
        print(f"  PATIENT  : {patient_name.upper()}")
        print(f"  STATUS   : REQUIRES IMMEDIATE ASSISTANCE")
        print(f"  TIME     : {time.strftime('%H:%M:%S')}")
        print("=" * 60)
        print("\n  >>> Press ENTER to acknowledge and stop the alarm <<<\n")

        # Flush stdout so the message appears immediately in any terminal
        import sys
        sys.stdout.flush()

        # Open the console input directly — works even when stdin is a pipe
        try:
            if sys.platform == "win32":
                # On Windows, open CONIN$ directly so it always reads from
                # the physical keyboard, regardless of how the process was launched
                with open("CONIN$", "r") as con:
                    con.readline()
            else:
                with open("/dev/tty", "r") as tty:
                    tty.readline()
        except Exception:
            # Fallback: standard input() — works when running task.py directly
            try:
                input()
            except Exception:
                time.sleep(30)  # last resort: wait 30s then auto-resolve

        self._stop_alarm.set()
        print(f"[EMERGENCY] Alarm acknowledged by staff at {time.strftime('%H:%M:%S')}")

    def trigger(self, patient_name: str, location: str,
                syth, db=None, patient_id: int = None,
                session_id: int = None, trigger_phrase: str = ""):
        """
        Full emergency trigger:
        1. Print alert to terminal
        2. Speak patient name and location via TTS
        3. Start continuous alarm in background thread
        4. Save to database
        5. Wait for staff to press Enter to stop alarm

        Args:
            patient_name:   Name of the patient
            location:       Ward / room name
            syth:           speechSynthesize instance for TTS announcement
            db:             DatabaseManager instance (optional)
            patient_id:     Patient DB id (optional)
            session_id:     Session DB id (optional)
            trigger_phrase: What the patient said that triggered the alert
        """

        # 1. Save to database immediately
        emergency_id = None
        if db and patient_id and session_id:
            emergency_id = db.save_emergency(
                patient_id=patient_id,
                session_id=session_id,
                patient_name=patient_name,
                trigger_phrase=trigger_phrase
            )

        # 2. Speak the alert via TTS
        alert_speech = (
            f"Attention. Attention. {patient_name} needs immediate assistance. "
            f"Please send staff to {location} immediately. "
            f"{patient_name} needs assistance now."
        )
        try:
            syth.play_audio(
                text=alert_speech,
                filename=f"emergency_{patient_name}.wav",
                playback_speed=1.0,
                is_synthesize=True
            )
        except Exception as e:
            print(f"[EMERGENCY] TTS failed: {e}")

        # 3. Start continuous alarm in background thread
        self._stop_alarm.clear()
        self._alarm_thread = threading.Thread(
            target=self._alarm_loop, daemon=True
        )
        self._alarm_thread.start()

        # 4. Block until staff acknowledge (presses Enter)
        self._wait_for_staff(patient_name)

        # 5. Resolve in database
        if db and emergency_id:
            db.resolve_emergency(emergency_id)

        # 6. Speak calm resolution message
        try:
            calm_text = f"Thank you. Help is on the way for {patient_name}."
            syth.play_audio(
                text=calm_text,
                filename=f"emergency_resolved_{patient_name}.wav",
                playback_speed=1.0,
                is_synthesize=True
            )
        except Exception as e:
            print(f"[EMERGENCY] TTS resolution failed: {e}")
