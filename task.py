import time
from abc import ABC, abstractmethod
from collections import defaultdict
import datetime

import torch

from speech_synthesis import speechSynthesize
from speech_Recognition import speechRecognizer
from classifier import Classifier

PLAY_SPEED = 1
WAIT_TIME = 1
SPEED = 1
SPEED_SLOW = 0.85

from enum import Enum, auto


class TaskState(Enum):
    INIT = auto()
    READY_CHECK = auto()
    WAITING_FOR_KEYWORD = auto()  # New state
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
    def __init__(self, id=0, name=None, instructions=None):
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
        self.speed=SPEED

    # Handle the special state in one function
    def _handle_state(self, state, syth):
        if state.get("stop") == 1:
            if self.rest_count < 3:
                self.rest(syth, 1)
                self.rest_count += 1
                return "retry"
            else:
                self.rest_count = 0
                return "exit"

        elif state.get("emergency") == 1:
            self.finish_task(syth)
            return "exit"

        elif state.get("dont_know") == 1:
            self.speed = SPEED_SLOW
            self.encourage(syth)
            return "retry"
        elif state.get("correct") == 0:
            syth.play_audio()
            return "retry"
        return "continue"

    # def _ask_until_done(self, question, syth, rcg, clf):
    #     while True:
    #         state, result = question.ask_question(syth, rcg, clf)
    #
    #         action = self._handle_state(state, syth)
    #         if action == "retry":
    #             continue
    #         if action == "exit":
    #             return "exit", None
    #
    #         return "answered", result

    def ready_check(self, syth: speechSynthesize, rcg: speechRecognizer, clf: Classifier):
        text = "If you are ready to have this task, Please said yes after hearing the beep."
        syth.play_audio(
            text=text,
            filename=f"ready_check.wav",
            playback_speed=1,
            is_synthesize=True
        )

    # The main cognitive rehabilitation task( Orientation task ) would be implemented in this function
    def perform_task(self, syth: speechSynthesize, rcg: speechRecognizer, clf: Classifier):

        print(self.instructions)
        intro = "Hi, my name is Jennet, your personal female medical assistant."
        syth.play_audio(
            text=intro,
            filename=f"introduce.wav",
            playback_speed=self.speed,
            is_synthesize=True
        )
        state = TaskState.INIT
        q_index = 0
        bd_index = 0
        current_question = None
        breakdown_list = []

        while state not in (TaskState.FINISH, TaskState.EXIT):

            # Initial
            if state == TaskState.INIT:
                if len(self.list) == 0:
                    state = TaskState.FINISH
                    continue

                q_index = 0
                state = TaskState.READY_CHECK
            # Ready check
            elif state == TaskState.READY_CHECK:
                check_result = self.ready_check(syth, rcg, clf)

                if check_result == "ready":
                    # Play instructions and proceed

                    syth.play_audio(
                        text=self.instructions,
                        filename=f"{self.name}_{self.id}_instruction.wav",
                        playback_speed=self.speed,
                        is_synthesize=True
                    )
                    state = TaskState.INTERACT_MAIN

                elif check_result == "waiting":
                    state = TaskState.WAITING_FOR_KEYWORD

                elif check_result == "exit":
                    state = TaskState.EXIT

            # Waiting for Keyword State
            elif state == TaskState.WAITING_FOR_KEYWORD:
                t = WAIT_TIME * 60
                keyword_result = self.wait_for_keyword(rcg, keyword="Emily", timeout=t*60)

                if keyword_result == "detected":
                    confirm_text = "I heard you! Let's start now."
                    syth.play_audio(
                        text=confirm_text,
                        filename="keyword_detected.wav",
                        playback_speed=self.speed,
                        is_synthesize=True
                    )
                    # Play instructions
                    syth.play_audio(
                        text=self.instructions,
                        filename=f"{self.name}_{self.id}_instruction.wav",
                        playback_speed=self.speed,
                        is_synthesize=True
                    )
                    state = TaskState.INTERACT_MAIN

                elif keyword_result == "timeout":
                    timeout_text = "I haven't heard from you. Let me check if you're ready again."
                    syth.play_audio(
                        text=timeout_text,
                        filename="keyword_timeout.wav",
                        playback_speed=self.speed,
                        is_synthesize=True
                    )
                    state = TaskState.READY_CHECK  # Go back to ready check

            # First, ask the main question
            elif state == TaskState.INTERACT_MAIN:
                current_question = self.list[q_index]

                q_state, result = current_question.ask_question(syth, rcg, clf, speed=self.speed)
                action = self._handle_state(q_state, syth)
                print(f"The state of main question is {q_state}\nThe result is {result}\n the action is {action}")

                if action == "exit":
                    state = TaskState.EXIT
                elif action == "retry":
                    continue
                elif action == "correct" or action == "continue":
                    print("Main question is correct")
                    state = TaskState.NEXT

                else:  # wrong
                    breakdown_list = current_question.list
                    bd_index = 0

                    if len(breakdown_list) == 0:
                        state = TaskState.NEXT
                    else:
                        state = TaskState.INTERACT_BREAKDOWN
            # If the main question is worry => ask breakdown question
            elif state == TaskState.INTERACT_BREAKDOWN:

                bd_question = breakdown_list[bd_index]

                q_state, result = bd_question.ask_question(syth, rcg, clf, speed=self.speed)

                decision = self._handle_state(q_state, syth)

                if decision == "exit":
                    state = TaskState.EXIT

                elif decision == "retry":
                    continue

                elif decision == "correct":
                    state = TaskState.NEXT

                else:  # still wrong
                    bd_index += 1
                    if bd_index >= len(breakdown_list):
                        state = TaskState.NEXT
                    else:
                        state = TaskState.INTERACT_BREAKDOWN

            # If the main question is correct => next questiion
            elif state == TaskState.NEXT:
                text = self.list[q_index].end_text
                if text is not None:
                    syth.play_audio(text=text,
                                    filename=f"main_q_{q_index}_end.wav",
                                    playback_speed=self.speed,
                                    is_synthesize=True)

                q_index += 1
                if q_index >= len(self.list):
                    state = TaskState.FINISH
                else:
                    state = TaskState.INTERACT_MAIN

        if state == TaskState.FINISH:
            self.finish_task(syth)
        elif state == TaskState.EXIT:
            print(" Emergency, we need some help here.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return 0

    def calculate(self):
        if len(self.list) == 0:
            return 0
        else:
            for i in self.list:
                self.score += i.score
            self.error_rate = self.score / len(list)
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
            self.thinking_time = total_time / len(list)

    def finish_task(self, syth: speechSynthesize):
        text = "That is the end of the task. See you."
        syth.play_audio(text=text,
                        filename=fr"break.wav",
                        is_synthesize=True,
                        playback_speed=self.speed)
        return 0

    def encourage(self, syth: speechSynthesize):

        if self.retry_count < 1:
            text = "Come on, Let's have another try"
            syth.play_audio(text=text,
                            filename=fr"encourage.wav",
                            is_synthesize=True,
                            playback_speed=self.speed)
        else:
            self.retry_count = 0
            text = "What about the next one."
            syth.play_audio(text=text,
                            filename=fr"next.wav",
                            is_synthesize=True,
                            playback_speed=self.speed)
            return 0

    def rest(self, syth: speechSynthesize, t: int):
        if self.rest_count < 3:
            duration = t * 60
            text = f"Let's have a {t} minutes rest."
            syth.play_audio(text=text,
                            filename=fr"rest_{t}.wav",
                            is_synthesize=True,
                            playback_speed=self.speed)
            time.sleep(duration)
        else:
            self.finish_task(syth)
            return 0

    def ready_check(self, syth: speechSynthesize, rcg: speechRecognizer, clf: Classifier):
        """
        Check if patient is ready to start.
        Returns: 'ready', 'waiting', or 'exit'
        """
        text = "If you are ready to start this task, please say yes after hearing the beep."
        syth.play_audio(
            text=text,
            filename=f"ready_check.wav",
            playback_speed=self.speed,
            is_synthesize=True
        )

        time.sleep(1)
        response, _, _ = rcg.record_audio(duration=5)

        # Check if patient said "yes"
        state, score = clf.state_match(input_text=response, target_text="yes")

        # Handle special states (emergency, stop)
        if state.get("emergency") == 1:
            return "exit"

        if state.get("stop") == 1:
            return "exit"

        # Patient is ready
        if state.get("correct") == 1:
            confirm_text = "Great! Let's begin."
            syth.play_audio(
                text=confirm_text,
                filename="ready_confirmed.wav",
                playback_speed=self.speed,
                is_synthesize=True
            )
            return "ready"

        # Patient said "no" or unclear response
        else:
            wait_text = f"No problem. I'll wait you for {WAIT_TIME} minute."
            syth.play_audio(
                text=wait_text,
                filename="waiting_for_keyword.wav",
                playback_speed=self.speed,
                is_synthesize=True
            )
            return "waiting"

    def wait_for_keyword(self, rcg: speechRecognizer, keyword="jennett", timeout=300):
        """
        Continuously monitor for keyword using VAD.
        Returns: 'detected' or 'timeout'
        """
        # print(f"Waiting for keyword '{keyword}'... (timeout: {timeout}s)")

        start_time = time.time()
        check_interval = 2 # Check every 0.5 seconds
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

    def ask_question(self, syth: speechSynthesize, rcg: speechRecognizer, clf: Classifier, speed=1):

        print(self.question)
        syth.play_audio(text=self.question,
                        filename=f"{self.name}_{self.id}_question.wav",
                        playback_speed=speed
                        )
        """
        this function would ask this question and detect if the answer is correct or incorrect
        """

        time.sleep(1)

        text, time_0, time_1 = rcg.record_audio()

        self.thinking_time = time_0
        state, score = clf.state_match(input_text=text, target_text=self.answer)
        self.state = state
        result = state["correct"]
        print(f"the result is {result}, input_text is P{text}, target test is {self.answer}")

        # If detect the correct answer -> correct response
        if result == 1:
            self.score = 1
            syth.play_audio(text=self.correct_response,
                            filename=f"{self.name}_{self.id}_correct_response.wav",
                            playback_speed=speed,
                            is_synthesize=True)
            self.state_list.append(state)
            return state, result
        # If detect don't detect the correct answer =>go through all the answer of sub-questions
        if len(self.list) > 0:

            matched = 0

            for sub_q in self.list:
                sub_state, _ = clf.state_match(input_text=text, target_text=sub_q.answer)
                if sub_state["correct"] == 1:
                    matched += 1

            if matched == len(self.list):
                final_state = dict(state)
                final_state["correct"] = 1

                syth.play_audio(
                    text=self.correct_response,
                    filename=f"{self.name}_{self.id}_correct_response.wav",
                    playback_speed=speed,
                    is_synthesize=True
                )
                self.state_list.append(final_state)

                return final_state, 1

            # ---- Case 3: incorrect ----
        syth.play_audio(
            text=self.incorrect_response,
            filename=f"{self.name}_{self.id}_incorrect_response.wav",
            playback_speed=speed,
            is_synthesize=True
        )

        print(f"the state is {state}")
        self.state_list.append(state)

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

    # Initial question and task
    t_0 = "What specific date of today? "
    t_00 = "what year is it this year? "
    t_01 = "What mouth is it this mouth? "
    t_02 = "Which day of this month is it today? "
    t_03 = "What day is it today in this week? "

    today = datetime.date.today()
    year = today.strftime("%Y")
    month = today.strftime("%B")
    day_int = today.day
    day_ordinal = ordinal(day_int)
    weekday = today.strftime("%A")

    print(f"{type(year)}_({year})")

    a_0 = f"{weekday}, {month} the {day_ordinal}, {year}"
    a_00 = year
    a_01 = month
    a_02 = f"{day_ordinal}"
    a_03 = weekday

    now = time.localtime()
    formatted_time = time.strftime("It is %I:%M %p on %A, %B %d, %Y.", now)
    instruction = f" my name is Jennet. {formatted_time}. There is an orientation task in your planner, try to answer the question after hearing the beep and avoid any guessing."

    print(f"{type(year)}...{year}...{type(month)}...{month}...{type(day_ordinal)}...{day_ordinal}...{type(weekday)}...{weekday}")
    end_text = f"Today is {a_0}. If you need to check what the date is, it is written on your Orientation Chart or you can ask any member of staff."

    p_0 = Patient(id=0, name="Jerry", address="Lennon studio")
    task_0 = Task(name="Orientation_task")
    task_0.set_instructions(instruction)
    q_main = Question(q_id=0, name="main_data", question=t_0, answer=a_0, is_synthesize=True, end_text=end_text)
    q_0 = Question(q_id=1, name="bd_0", question=t_00, answer=a_00, is_synthesize=True)
    q_1 = Question(q_id=2, name="bd_1", question=t_01, answer=a_01, is_synthesize=True)
    q_2 = Question(q_id=3, name="bd_3", question=t_02, answer=a_02, is_synthesize=True)
    q_3 = Question(q_id=4, name="bd_4", question=t_03, answer=a_03, is_synthesize=True)

    # list_bd = [q_0, q_1, q_2, q_3]
    list_bd = [q_1]

    q_main.set_list(list_bd)
    list_q = [q_main]
    task_0.set_list(list_q)

    sr = speechRecognizer(model_name="faster_whisper")
    tts = speechSynthesize(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    classifier_0 = Classifier()

    task_0.perform_task(tts, sr, classifier_0)
