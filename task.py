import time
from abc import ABC, abstractmethod
from collections import defaultdict
import datetime

import torch

from speech_synthesis import speechSynthesize
from speech_Recognition import speechRecognizer
from classifier import Classifier
from database.db import DatabaseManager
from emergency import EmergencyAlert
from report import generate_report

PLAY_SPEED = 1
WAIT_TIME = 1
SPEED = 1
SPEED_SLOW = 0.85
REST_TIME = 3
CHANGEABLE = True
UNCHANGEABLE = True
MAX_RETRY_PER_QUESTION = 1

from enum import Enum, auto


class TaskState(Enum):
    INIT = auto()
    READY_CHECK = auto()
    WAITING_FOR_KEYWORD = auto()
    INTERACT_MAIN = auto()
    INTERACT_BREAKDOWN = auto()
    NEXT = auto()
    FINISH = auto()
    EXIT = auto()


class Item(ABC):
    def __init__(self, id=0, name=None):
        self.list = []
        self.id = id
        self.name = name
        self._dic = {}

    def set_list(self, input_list):
        if type(input_list) is not list:
            raise TypeError(f"Expected list, got {type(input_list).__name__}")

        if len(input_list) == 0:
            raise ValueError("Input list should not be empty")
        if type(input_list) is list:
            n_list = input_list
            self.list = n_list
        else:
            print("111")
            raise ValueError("input value should be a list and should not be an empty list.")
        return 0

    def get_list(self):
        list_n = self.list
        return list_n

    def set_dic(self, key, value):
        self._dic[key] = value
        return 0

    def get_dic(self):
        dic_n = self._dic
        return dic_n

    @abstractmethod
    def logging(self):
        pass


class Patient(Item):
    def __init__(self, id=0, name=None, address=None):
        super().__init__(id, name)
        self._address = address

    def logging(self):
        return 0


class Task(Item):
    def __init__(self, id=0, name=None, instructions=None, time=None):
        super().__init__(id, name)
        self._last_result = None
        self.date = None
        self.score = 0
        self.thinking_time = 0
        self.error_rate = 0.0
        self.instructions = instructions
        self.speed = 1
        self.rest_count = 0
        self.retry_count = 0
        self.speed = SPEED
        self.time = time
        self.patient_name = ""
        self.location = "the ward"

    # Handle the special state in one function
    def _handle_state(self, state, syth, current_state: TaskState = None):
        if state.get("stop") == 1:
            if self.rest_count < 3:
                self.rest(syth, 1)
                self.rest_count += 1
                return "return"
            else:
                self.rest_count = 0
                return "exit"

        elif state.get("emergency") == 1:
            self.finish_task(syth)
            return "exit"
            self.speed = SPEED_SLOW
            print(f"NEW SPEED : {self.speed}")
            decision = self.encourage(syth)
            return decision

        elif state.get("repeat") == 1:
            return "repeat"

        elif state.get("require") == 1:
            return "require"

        elif state.get("silence") == 1:
            return "silence"

        elif state.get("correct") == 0:
            if current_state == TaskState.INTERACT_MAIN:
                print("NO RETRY ON MAIN QUESTION")
                return "continue"
            return "retry"
        elif state.get("correct") == 1:
            return "correct"
        return "continue"

    def _force_advance(self, state, bd_index, breakdown_list):
        """
        When retry budget exhausted, decide how FSM should move.
        """

        if state == TaskState.INTERACT_MAIN:
            return TaskState.INTERACT_BREAKDOWN

        elif state == TaskState.INTERACT_BREAKDOWN:
            bd_index += 1
            if bd_index >= len(breakdown_list):
                return TaskState.NEXT
            else:
                return TaskState.INTERACT_BREAKDOWN

    # The main cognitive rehabilitation task( Orientation task ) would be implemented in this function
    def perform_task(self, syth: speechSynthesize, rcg: speechRecognizer,
                     clf: Classifier, db: DatabaseManager = None,
                     session_id: int = None, patient_id: int = None,
                     emergency_alert: EmergencyAlert = None):

        print(self.instructions)
        state = TaskState.INIT
        q_index = 0
        bd_index = 0
        current_question = None
        breakdown_list = []
        pre_state = None
        retry_count = 0
        silence_count = 0    # global across all questions — 3 total silences ends session
        MAX_SILENCE = 3
        conv_history = []   # stores last exchanges for context-aware classification
        wrong_streak = 0    # consecutive wrong answers for adaptive encouragement

        while state not in (TaskState.FINISH, TaskState.EXIT):

            # Initial
            if state == TaskState.INIT:
                if len(self.list) == 0:
                    state = TaskState.FINISH
                    continue
                q_index = 0
                intro = f"Hello {self.patient_name} ,  my name is  Jennet , your personal female medical assistant."
                syth.play_audio(
                    text=intro,
                    filename=f"introduce_{self.speed}.wav",
                    playback_speed=self.speed,
                    is_synthesize=UNCHANGEABLE
                )
                syth.play_audio(
                    text=self.instructions,
                    filename=f"{self.name}_{self.id}_{self.speed}_instruction.wav",
                    playback_speed=self.speed,
                    is_synthesize=UNCHANGEABLE
                )

                state = TaskState.READY_CHECK
            # Ready check
            elif state == TaskState.READY_CHECK:
                check_result = self.ready_check(syth, rcg, clf)

                if check_result == "ready":
                    now = time.localtime()
                    formatted_time = time.strftime("It is  %I:%M  %p  on %A ,  %B %d , %Y.", now)
                    syth.play_audio(
                        text=formatted_time,
                        filename=f"{self.name}_{self.id}_{self.speed}_formatted_time.wav",
                        playback_speed=self.speed,
                        is_synthesize=CHANGEABLE
                    )
                    state = TaskState.INTERACT_MAIN

                elif check_result == "waiting":
                    state = TaskState.WAITING_FOR_KEYWORD

                elif check_result == "exit":
                    state = TaskState.EXIT

            # Waiting for Keyword State
            elif state == TaskState.WAITING_FOR_KEYWORD:
                t = WAIT_TIME
                keyword_result = self.wait_for_keyword(rcg, keyword="Jennet", timeout=t * 60)

                if keyword_result == "detected":
                    confirm_text = "I heard you! Let's start now."
                    syth.play_audio(
                        text=confirm_text,
                        filename=f"keyword_detected_{self.speed}.wav",
                        playback_speed=self.speed,
                        is_synthesize=UNCHANGEABLE
                    )
                    # Play instructions
                    syth.play_audio(
                        text=self.instructions,
                        filename=f"{self.name}_{self.id}_{self.speed}_instruction.wav",
                        playback_speed=self.speed,
                        is_synthesize=UNCHANGEABLE
                    )
                    state = TaskState.INTERACT_MAIN

                elif keyword_result == "timeout":
                    timeout_text = " Let me check if you're ready again."
                    syth.play_audio(
                        text=timeout_text,
                        filename=f"keyword_timeout_{self.speed}.wav",
                        playback_speed=self.speed,
                        is_synthesize=UNCHANGEABLE
                    )
                    state = TaskState.READY_CHECK  # Go back to ready check

            # First, ask the main question
            elif state == TaskState.INTERACT_MAIN:
                current_question = self.list[q_index]

                q_state, result = current_question.ask_question(
                    syth, rcg, clf, speed=self.speed,
                    db=db, session_id=session_id,
                    conv_history=conv_history
                )
                action = self._handle_state(q_state, syth, state)
                print(f"The state of main question is {q_state}\nThe result is {result}\n the action is {action}")

                if action == "exit":
                    state = TaskState.EXIT
                elif action == "repeat":
                    syth.play_audio(
                        text=f"Of course. {current_question.question}",
                        filename=f"repeat_main_{q_index}_{self.speed}.wav",
                        playback_speed=self.speed,
                        is_synthesize=UNCHANGEABLE
                    )
                    continue
                elif action == "require":
                    syth.play_audio(
                        text="I'm sorry, I am just an agent. Please ask a member of staff for assistance.",
                        filename=f"require_main_{self.speed}.wav",
                        playback_speed=self.speed,
                        is_synthesize=UNCHANGEABLE
                    )
                    continue
                elif action == "silence":
                    silence_count += 1
                    print(f"[SILENCE] Count: {silence_count}/{MAX_SILENCE}")
                    if silence_count >= MAX_SILENCE:
                        print("[SILENCE] Prolonged silence — ending session.")
                        syth.play_audio(
                            text=f"I will end the orientation now as you have not responded for a while. Goodbye {self.patient_name}.",
                            filename=f"silence_exit_{self.speed}.wav",
                            playback_speed=self.speed,
                            is_synthesize=UNCHANGEABLE
                        )
                        state = TaskState.EXIT
                    else:
                        syth.play_audio(
                            text="I did not hear you. Could you please answer the question?",
                            filename=f"silence_retry_{silence_count}_{self.speed}.wav",
                            playback_speed=self.speed,
                            is_synthesize=UNCHANGEABLE
                        )
                        continue
                elif action == "retry_dont_know":
                    continue
                elif action == "return":
                    text = "Let's back to our task."
                    syth.play_audio(text=text,
                                    filename=f"return_{self.speed}.wav",
                                    playback_speed=self.speed,
                                    is_synthesize=UNCHANGEABLE)
                elif action == "retry":
                    text = "Let's have another try."
                    syth.play_audio(text=text,
                                    filename=f"bridge_{self.speed}.wav",
                                    playback_speed=self.speed,
                                    is_synthesize=UNCHANGEABLE)
                    continue
                elif action == "correct":
                    print("Main question is correct")
                    silence_count = 0
                    wrong_streak  = 0
                    current_question.score = 1
                    conv_history.append({
                        "question": current_question.question,
                        "answer": current_question.answer,
                    })
                    if len(conv_history) > 2:
                        conv_history.pop(0)
                    state = TaskState.NEXT

                else:  # wrong — go to breakdown
                    wrong_streak += 1
                    self._adaptive_encouragement(syth, wrong_streak)
                    breakdown_list = current_question.list
                    bd_index = 0
                    bd_correct_count = 0

                    if len(breakdown_list) == 0:
                        state = TaskState.NEXT
                    else:
                        text_bd = "Let's break it down"
                        syth.play_audio(text=text_bd,
                                        filename=f"breakdown_bridge.wav",
                                        playback_speed=self.speed,
                                        is_synthesize=UNCHANGEABLE)
                        bd_retry_used = False
                        state = TaskState.INTERACT_BREAKDOWN

            # If the main question is worry => ask breakdown question
            elif state == TaskState.INTERACT_BREAKDOWN:

                bd_question = breakdown_list[bd_index]
                q_state, result = bd_question.ask_question(
                    syth, rcg, clf, speed=self.speed,
                    db=db, session_id=session_id,
                    conv_history=conv_history
                )
                decision = self._handle_state(q_state, syth, state)

                if decision == "exit":
                    state = TaskState.EXIT

                elif decision == "repeat":
                    syth.play_audio(
                        text=f"Of course. {bd_question.question}",
                        filename=f"repeat_bd_{bd_index}_{self.speed}.wav",
                        playback_speed=self.speed,
                        is_synthesize=UNCHANGEABLE
                    )
                    continue
                elif decision == "require":
                    syth.play_audio(
                        text="I'm sorry, I am just an agent. Please ask a member of staff for assistance.",
                        filename=f"require_bd_{self.speed}.wav",
                        playback_speed=self.speed,
                        is_synthesize=UNCHANGEABLE
                    )
                    continue

                elif decision == "silence":
                    silence_count += 1
                    print(f"[SILENCE] Count: {silence_count}/{MAX_SILENCE}")
                    if silence_count >= MAX_SILENCE:
                        print("[SILENCE] Prolonged silence — ending session.")
                        syth.play_audio(
                            text=f"I will end the orientation now as you have not responded for a while. Goodbye {self.patient_name}.",
                            filename=f"silence_exit_{self.speed}.wav",
                            playback_speed=self.speed,
                            is_synthesize=UNCHANGEABLE
                        )
                        state = TaskState.EXIT
                    else:
                        syth.play_audio(
                            text="I did not hear you. Could you please answer the question?",
                            filename=f"silence_retry_{silence_count}_{self.speed}.wav",
                            playback_speed=self.speed,
                            is_synthesize=UNCHANGEABLE
                        )
                        continue  # re-ask the same breakdown question

                elif decision == "return":
                    text = "Let's back to our task."
                    syth.play_audio(text=text,
                                    filename=f"return_{self.speed}.wav",
                                    playback_speed=self.speed,
                                    is_synthesize=UNCHANGEABLE)
                elif decision == "retry_dont_know":
                    continue
                elif decision == "retry":
                    if not bd_retry_used:
                        bd_retry_used = True
                        if decision == "retry_dont_know":
                            bd_retry_used = True
                        if decision == "retry":
                            text = "Let's try that once more."
                            syth.play_audio(text=text,

                                            filename=f"bd_retry_once_{self.speed}.wav",

                                            playback_speed=self.speed,

                                            is_synthesize=UNCHANGEABLE)

                        continue  # Back to current breakdown
                    else:
                        # After retry → Next breakdown
                        bd_retry_used = False
                        bd_index += 1
                        if bd_index >= len(breakdown_list):
                            print("NO MORE BREAKDOWN")
                            state = TaskState.NEXT
                        else:
                            state = TaskState.INTERACT_BREAKDOWN
                            text = "Let's move on."
                            syth.play_audio(text=text,
                                            filename=f"bd_move_on_{self.speed}.wav",
                                            playback_speed=self.speed,
                                            is_synthesize=UNCHANGEABLE)
                else:  # correct or continue — advance breakdown
                    wrong_streak = 0
                    bd_correct_count += 1
                    bd_retry_used = False
                    bd_index += 1
                    retry_count = 0
                    if bd_index >= len(breakdown_list):
                        # Award main question score if all breakdowns answered correctly
                        if bd_correct_count == len(breakdown_list):
                            current_question.score = 1
                            print("[SCORE] All breakdown questions correct — main question scored")
                        state = TaskState.NEXT
                    else:
                        state = TaskState.INTERACT_BREAKDOWN

            # If the main question is correct => next questiion
            elif state == TaskState.NEXT:
                q_index += 1
                if q_index >= len(self.list):
                    q_index -= 1
                    text = self.list[q_index].end_text
                    if text is not None:
                        syth.play_audio(text=text,
                                        filename=f"main_q_{q_index}_{self.speed}_end.wav",
                                        playback_speed=self.speed,
                                        is_synthesize=UNCHANGEABLE)
                    state = TaskState.FINISH
                else:
                    state = TaskState.INTERACT_MAIN

        # The end of the task =>  logging and
        if state == TaskState.FINISH:
            self.finish_task(syth)
        elif state == TaskState.EXIT:
            print(f"[EMERGENCY] {self.patient_name} needs assistance!")
            if emergency_alert:
                emergency_alert.trigger(
                    patient_name=self.patient_name,
                    location=self.location,
                    syth=syth,
                    db=db,
                    patient_id=patient_id,
                    session_id=session_id,
                    trigger_phrase="Emergency detected during session"
                )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return 0

    def calculate(self):
        if len(self.list) == 0:
            return 0
        else:
            for i in self.list:
                self.score += i.score
            self.error_rate = self.score / len(self.list)
            return 0

    def set_instructions(self, i_text):
        self.instructions = i_text
        return 0

    def set_thinking_time(self, time=None):
        if len(self.list) == 0:
            self.thinking_time = time
        else:
            total_time = 0.0
            for i in self.list:
                total_time += i.thingking_time
            self.thinking_time = total_time / len(self.list)

    def finish_task(self, syth: speechSynthesize):
        text = "That is the end of the task. See you."
        syth.play_audio(text=text,
                        filename=fr"break_{self.speed}.wav",
                        is_synthesize=UNCHANGEABLE,
                        playback_speed=self.speed)
        return 0

    def encourage(self, syth: speechSynthesize):

        if self.retry_count < 1:
            text = "Don't worry, I would slow down a little bit, Let's have another try。 "
            syth.play_audio(text=text,
                            filename=fr"encourage_{self.speed}.wav",
                            is_synthesize=UNCHANGEABLE,
                            playback_speed=self.speed)
            return "retry_dont_know"
        else:
            self.retry_count = 0
            # text = "What about the next one."
            # syth.play_audio(text=text,
            #                 filename=fr"next_{self.speed}.wav",
            #                 is_synthesize=UNCHANGEABLE,
            #                 playback_speed=self.speed)
            return "continue"

    def rest(self, syth: speechSynthesize, t: int):
        if self.rest_count < 3:
            duration = t * 60
            text_0 = f"Let's have a {t} minutes rest."
            syth.play_audio(text=text_0,
                            filename=fr"rest_{t}_{self.speed}.wav",
                            is_synthesize=CHANGEABLE,
                            playback_speed=self.speed)
            time.sleep(duration)
            text_1 = "Let's go back to the task."
            syth.play_audio(text=text_1,
                            filename=fr"rest_{t}_{self.speed}.wav",
                            is_synthesize=UNCHANGEABLE,
                            playback_speed=self.speed)

        else:
            self.finish_task(syth)
            return 0

    def ready_check(self, syth: speechSynthesize, rcg: speechRecognizer, clf: Classifier):
        """
        Check if patient is ready to start.
        Returns: 'ready', 'waiting', 'retry', or 'exit'
        After 3 consecutive silences, ends the session.
        """
        silence_count = 0
        MAX_READY_SILENCE = 3

        while True:
            text = "If you are ready  to start this task  ,  please say yes  after hearing the beep."
            syth.play_audio(
                text=text,
                filename=f"ready_check_{self.speed}.wav",
                playback_speed=self.speed,
                is_synthesize=UNCHANGEABLE
            )

            time.sleep(1)
            response, _, _ = rcg.record_audio(duration=5)

            # Empty transcription — treat as silence
            if not response or response.strip() == "":
                silence_count += 1
                print(f"[READY_CHECK] Silence {silence_count}/{MAX_READY_SILENCE}")
                if silence_count >= MAX_READY_SILENCE:
                    syth.play_audio(
                        text=f"I will end the orientation now as you do not appear to be ready. Goodbye {self.patient_name}.",
                        filename=f"ready_silence_exit_{self.speed}.wav",
                        playback_speed=self.speed,
                        is_synthesize=UNCHANGEABLE
                    )
                    return "exit"
                syth.play_audio(
                    text="Sorry, I did not hear you. Please say yes if you are ready.",
                    filename=f"ready_check_retry_{self.speed}.wav",
                    playback_speed=self.speed,
                    is_synthesize=UNCHANGEABLE
                )
                continue

            # Got a response — classify it
            state, score, _ = clf.state_match(input_text=response, target_text="yes")

            if state.get("emergency") == 1:
                return "exit"

            if state.get("correct") == 1:
                confirm_text = "Great, Let's begin."
                syth.play_audio(
                    text=confirm_text,
                    filename=f"ready_confirmed_{self.speed}.wav",
                    playback_speed=self.speed,
                    is_synthesize=UNCHANGEABLE
                )
                return "ready"

            elif clf.is_negative(response):
                wait_text = f"No problem. I'll wait for {WAIT_TIME} minute."
                syth.play_audio(
                    text=wait_text,
                    filename=f"waiting_for_keyword_{self.speed}.wav",
                    playback_speed=self.speed,
                    is_synthesize=CHANGEABLE
                )
                return "waiting"

            else:
                # Free talk or unrecognised — treat same as silence in ready check
                silence_count += 1
                print(f"[READY_CHECK] Free talk treated as silence {silence_count}/{MAX_READY_SILENCE}")
                if silence_count >= MAX_READY_SILENCE:
                    syth.play_audio(
                        text=f"I will end the orientation now as you do not appear to be ready. Goodbye {self.patient_name}.",
                        filename=f"ready_silence_exit_{self.speed}.wav",
                        playback_speed=self.speed,
                        is_synthesize=UNCHANGEABLE
                    )
                    return "exit"
                syth.play_audio(
                    text="If you are ready, please say yes.",
                    filename=f"ready_check_freetalk_{self.speed}.wav",
                    playback_speed=self.speed,
                    is_synthesize=UNCHANGEABLE
                )

    def wait_for_keyword(self, rcg: speechRecognizer, keyword="jennett", timeout=300):
        """
        Continuously monitor for keyword using VAD.
        Returns: 'detected' or 'timeout'
        """
        # print(f"Waiting for keyword '{keyword}'... (timeout: {timeout}s)")

        start_time = time.time()
        check_interval = 2  # Check every 0.5 seconds
        print(f" Sleep for {timeout}s...")
        time.sleep(timeout)
        # while (time.time() - start_time) < timeout:

        # Use keyword detection with short timeout
        #     detected = rcg.listen_keyword(
        #         keywords=[keyword.lower()],
        #         timeout=check_interval
        #     )
        #
        #     if detected:
        #         print(f"✓ Keyword '{keyword}' detected!")
        #         return "detected"
        #
        #     time.sleep(0.1)  # Small delay between checks
        #
        # print(f"⏱ Keyword detection timeout after {timeout}s")
        return "timeout"

    def _adaptive_encouragement(self, syth: speechSynthesize, wrong_streak: int):
        """
        Play encouragement that adapts based on consecutive wrong answers.
        Also slows Jennet down after 3+ wrong in a row.
        """
        if wrong_streak == 1:
            text = "That's okay. Let's try the next one."
            filename = f"encourage_1_{self.speed}.wav"
        elif wrong_streak == 2:
            text = f"You're doing well {self.patient_name}, don't worry. Let's keep going."
            filename = f"encourage_2_{self.speed}.wav"
        else:
            # 3+ wrong in a row — slow down
            if self.speed != SPEED_SLOW:
                self.speed = SPEED_SLOW
                print(f"[ADAPTIVE] 3 wrong in a row — slowing to {SPEED_SLOW}")
            text = (f"Let me speak a little slower for you {self.patient_name}. "
                    f"Take your time, there is no rush.")
            filename = f"encourage_3_{self.speed}.wav"

        syth.play_audio(
            text=text,
            filename=filename,
            playback_speed=self.speed,
            is_synthesize=UNCHANGEABLE
        )

    def change_speed(self):
        return

    def logging(self):
        return 0


class Question(Item):
    def __init__(self, answer, q_id=0, name=None, question=None, is_synthesize=True, end_text=None):
        super().__init__(q_id, name)
        self.date = None
        self.answer = answer
        self.score = 0
        self.thinking_time = 0.0
        self.avg_thinking_time = 0.0
        self.error_rate = 0.0
        struct_q = f"Could you tell me {question} after hearing the beep? "
        self.question = struct_q
        self.hint = None
        self.correct_response = f"Great, the answer is {self.answer}"
        self.incorrect_response = f"Don't worry, the answer is {self.answer}"
        self.is_synthesize = is_synthesize
        self.state_list = []
        self.end_text = end_text
        self.speaking_time = 0

    def _generate_hint(self) -> str:
        """
        Generate a gentle one-line hint for the answer.
        - For short answers (1 word, ≤6 chars): give the first letter
        - For longer answers: give the first word
        """
        answer = str(self.answer).strip()
        words  = answer.split()
        if len(words) == 1 and len(answer) <= 6:
            return f"It starts with the letter {answer[0].upper()}."
        elif len(words) == 1:
            return f"The answer starts with {answer[:2]}..."
        else:
            return f"It begins with {words[0]}."

    def calculate(self):
        if len(self.list) == 0:
            return 0
        else:
            for i in self.list:
                self.score += i.score
            self.error_rate = self.score / len(list)
            return 0

    def set_avg_thinking_time(self, t_time: float):
        if len(self.list) == 0:
            self.thinking_time = t_time
        else:
            total_time = 0.0
            for i in self.list:
                total_time += i.thingking_time
            self.avg_thinking_time = total_time / len(list)

    def ask_question(self, syth: speechSynthesize, rcg: speechRecognizer,
                     clf: Classifier, speed=1, db=None, session_id=None,
                     conv_history: list = None):

        print(self.question)
        syth.play_audio(text=self.question,
                        filename=f"{self.name}_{self.id}_{speed}_question.wav",
                        playback_speed=speed
                        )
        """
        this function would ask this question and detect if the answer is correct or incorrect
        """

        time.sleep(1)

        text, time_0, time_1 = rcg.record_audio()

        self.thinking_time = time_0
        self.speaking_time = time_1 - time_0
        t_0 = time.time()
        state, score, confidence = clf.state_match(
            input_text=text,
            target_text=self.answer,
            context=conv_history
        )
        t_1 = time.time()
        print(f"The processing time of classifier is {t_0 - t_1}")
        print(f"Confidence: {confidence:.2%}")
        result = state["correct"]
        print(f"the result is {result}, input_text is P{text}, target test is {self.answer}")

        # If detect the correct answer -> correct response
        if score == 2:
            print("Special state detected...")
            if result == 1:
                self.score = 1
            # Save response to database
            if db and session_id:
                db.save_response(
                    session_id=session_id,
                    question_name=self.name,
                    question_text=self.question,
                    expected_answer=self.answer,
                    patient_answer=text,
                    is_correct=result,
                    thinking_time=self.thinking_time,
                    speaking_time=self.speaking_time,
                    state=state,
                    confidence=confidence,
                )
            return state, result
        if result == 1:
            self.score = 1
            syth.play_audio(text=self.correct_response,
                            filename=f"{self.name}_{self.id}_{speed}_correct_response.wav",
                            playback_speed=speed,
                            is_synthesize=CHANGEABLE)
            self.state_list.append(state)
            # Save response to database
            if db and session_id:
                db.save_response(
                    session_id=session_id,
                    question_name=self.name,
                    question_text=self.question,
                    expected_answer=self.answer,
                    patient_answer=text,
                    is_correct=result,
                    thinking_time=self.thinking_time,
                    speaking_time=self.speaking_time,
                    state=state,
                    confidence=confidence,
                )
            return state, result
        # If detect don't detect the correct answer => go through all the answer of sub-questions
        if len(self.list) > 0:

            matched = 0

            for sub_q in self.list:
                sub_state, _, _ = clf.state_match(
                    input_text=text,
                    target_text=sub_q.answer,
                    context=conv_history
                )
                if sub_state["correct"] == 1:
                    matched += 1

            if matched == len(self.list):
                final_state = dict(state)
                final_state["correct"] = 1

                syth.play_audio(
                    text=self.correct_response,
                    filename=f"{self.name}_{self.id}_{speed}_correct_response.wav",
                    playback_speed=speed,
                    is_synthesize=CHANGEABLE
                )
                self.state_list.append(final_state)
                # Save response to database
                if db and session_id:
                    db.save_response(
                        session_id=session_id,
                        question_name=self.name,
                        question_text=self.question,
                        expected_answer=self.answer,
                        patient_answer=text,
                        is_correct=1,
                        thinking_time=self.thinking_time,
                        speaking_time=self.speaking_time,
                        state=final_state,
                        confidence=confidence,
                    )
                return final_state, 1

            # ---- Case 3: incorrect — give hint first, then reveal ----
        # Step 1: hint
        hint_text = self._generate_hint()
        syth.play_audio(
            text=f"Not quite. Here is a clue — {hint_text}",
            filename=f"{self.name}_{self.id}_{speed}_hint.wav",
            playback_speed=speed,
            is_synthesize=CHANGEABLE
        )

        # Step 2: give patient one more attempt after the hint
        time.sleep(1)
        hint_text2, hint_t0, hint_t1 = rcg.record_audio()
        hint_state, hint_score, hint_conf = clf.state_match(
            input_text=hint_text2,
            target_text=self.answer,
            context=conv_history
        )

        # Emergency during hint attempt — propagate immediately
        if hint_state.get("emergency") == 1 or hint_state.get("stop") == 1:
            return hint_state, 2

        if hint_state.get("correct") == 1:
            self.score = 1
            syth.play_audio(
                text=self.correct_response,
                filename=f"{self.name}_{self.id}_{speed}_correct_response.wav",
                playback_speed=speed,
                is_synthesize=CHANGEABLE
            )
            if db and session_id:
                db.save_response(
                    session_id=session_id,
                    question_name=self.name,
                    question_text=self.question,
                    expected_answer=self.answer,
                    patient_answer=hint_text2,
                    is_correct=1,
                    thinking_time=hint_t0,
                    speaking_time=hint_t1 - hint_t0,
                    state=hint_state,
                    confidence=hint_conf,
                )
            return hint_state, 1

        # Step 3: reveal full answer
        syth.play_audio(
            text=self.incorrect_response,
            filename=f"{self.name}_{self.id}_{speed}_incorrect_response.wav",
            playback_speed=speed,
            is_synthesize=CHANGEABLE
        )

        print(f"the state is {state}")
        self.state_list.append(state)
        # Save response to database
        if db and session_id:
            db.save_response(
                session_id=session_id,
                question_name=self.name,
                question_text=self.question,
                expected_answer=self.answer,
                patient_answer=text,
                is_correct=result,
                thinking_time=self.thinking_time,
                speaking_time=self.speaking_time,
                state=state,
                confidence=confidence,
            )

        return state, result

    def logging(self):
        return 0


def ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


if __name__ == "__main__":

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Staff enters patient name and location ─────────────────────────
    PATIENT_NAME = input("Enter patient name: ").strip()
    LOCATION     = input("Enter patient location (e.g. Ward 3, Room 5): ").strip()

    # ── Initialise database ────────────────────────────────────────────
    db = DatabaseManager()
    patient = db.get_or_create_patient(name=PATIENT_NAME)
    print(f"[DB] Patient: {patient['name']} | First seen: {patient['created_at']}")

    history = db.get_patient_history(PATIENT_NAME)
    if history:
        print(f"[DB] {PATIENT_NAME} has {len(history)} previous session(s).")
        last = history[0]
        print(f"[DB] Last session: {last['started_at']} | Score: {last['total_score']}/{last['max_score']}")
    else:
        print(f"[DB] First session for {PATIENT_NAME}.")

    # ── Build questions ────────────────────────────────────────────────
    t_0  = " What the date is today? "
    t_00 = " what is the year? "
    t_01 = " What is the month? "
    t_02 = " What is the date today? "
    t_03 = " What day is it today in this week? "

    today       = datetime.date.today()
    year        = today.strftime("%Y")
    month       = today.strftime("%B")
    day_int     = today.day
    day_ordinal = ordinal(day_int)
    weekday     = today.strftime("%A")

    print(f"{type(year)}_({year})")

    a_0  = f"{weekday} {month} {day_int} {year}"
    a_00 = year
    a_01 = month
    a_02 = f"{day_int}"
    a_03 = weekday

    now = time.localtime()
    formatted_time = time.strftime("It is %I:%M %p on %A, %B %d, %Y.", now)
    instruction = " There is  an orientation task  in your planner,  try to answer the question  after hearing the beep."

    print(f"{type(year)}...{year}...{type(month)}...{month}...{type(day_ordinal)}...{day_ordinal}...{type(weekday)}...{weekday}")
    end_text = f"Today is {a_0}. If you need to check what the date is, it is written on your Orientation Chart or you can ask any member of staff."

    p_0    = Patient(id=0, name=PATIENT_NAME, address=LOCATION)
    task_0 = Task(name="Orientation_task")
    task_0.set_instructions(instruction)
    task_0.patient_name = PATIENT_NAME
    task_0.location     = LOCATION

    q_main = Question(q_id=0, name="main_date",  question=t_0,  answer=a_0,  is_synthesize=True, end_text=end_text)
    q_0    = Question(q_id=1, name="bd_year",    question=t_00, answer=a_00, is_synthesize=True)
    q_1    = Question(q_id=2, name="bd_month",   question=t_01, answer=a_01, is_synthesize=True)
    q_2    = Question(q_id=3, name="bd_day",     question=t_02, answer=a_02, is_synthesize=True)
    q_3    = Question(q_id=4, name="bd_weekday", question=t_03, answer=a_03, is_synthesize=True)

    list_bd = [q_0, q_1, q_2, q_3]
    q_main.set_list(list_bd)
    list_q = [q_main]
    task_0.set_list(list_q)

    # ── Start session in database ──────────────────────────────────────
    session_id = db.start_session(
        patient_id=patient["id"],
        task_name="Orientation_task"
    )

    # ── Initialise models and emergency alert ──────────────────────────
    sr            = speechRecognizer(model_name="faster_whisper", device="cuda",
                                      denoise=True)
    tts           = speechSynthesize(model_name="kokoro", voice="af_bella",
                                     lang_code="a", gpu=True)
    classifier_0  = Classifier()
    alert         = EmergencyAlert()

    # ── Dynamic difficulty — adjust speed based on last session ────────
    if history:
        last        = history[0]
        last_score  = last.get("total_score", 0)
        last_max    = last.get("max_score", 1) or 1
        last_pct    = last_score / last_max

        if last_pct == 0.0:
            task_0.speed = SPEED_SLOW
            instruction  = ("There is an orientation task in your planner. "
                            "I will speak slowly and clearly. "
                            "Try to answer each question after hearing the beep. "
                            "Take all the time you need.")
            print(f"[DIFFICULTY] Last score {last_score}/{last_max} (0%) "
                  f"— slow speed, simplified instructions")
        elif last_pct < 0.5:
            task_0.speed = SPEED_SLOW
            instruction  = ("There is an orientation task in your planner. "
                            "Try to answer each question after hearing the beep.")
            print(f"[DIFFICULTY] Last score {last_score}/{last_max} ({last_pct:.0%}) "
                  f"— slow speed")
        else:
            print(f"[DIFFICULTY] Last score {last_score}/{last_max} ({last_pct:.0%}) "
                  f"— normal speed")
        task_0.set_instructions(instruction)
    else:
        print("[DIFFICULTY] First session — normal speed")

    # ── Run task ───────────────────────────────────────────────────────
    task_0.perform_task(
        tts, sr, classifier_0,
        db=db,
        session_id=session_id,
        patient_id=patient["id"],
        emergency_alert=alert
    )

    # ── Close session ──────────────────────────────────────────────────
    total_score = sum(q.score for q in list_q)
    max_score   = len(list_q)
    db.end_session(
        session_id=session_id,
        total_score=total_score,
        max_score=max_score,
        completed=True
    )

    # ── Print session summary ──────────────────────────────────────────
    summary = db.get_session_summary(session_id)
    print("\n── Session Summary ───────────────────────────────────")
    print(f"Patient  : {summary['session']['patient_name']}")
    print(f"Task     : {summary['session']['task_name']}")
    print(f"Started  : {summary['session']['started_at']}")
    print(f"Ended    : {summary['session']['ended_at']}")
    print(f"Score    : {summary['session']['total_score']}/{summary['session']['max_score']}")
    print(f"Responses: {len(summary['responses'])}")
    print("──────────────────────────────────────────────────────\n")

    # ── Auto-generate clinical PDF report ─────────────────────────────
    try:
        report_path = generate_report(summary, db, output_dir="reports")
        print(f"[REPORT] Report ready: {report_path}")
    except Exception as e:
        print(f"[REPORT] Could not generate report: {e}")
