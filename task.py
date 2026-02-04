import time
from abc import ABC, abstractmethod
from collections import defaultdict
import datetime
from speech_synthesis import speechSynthesize
from speech_Recognition import speechRecognizer
from classifier import Classifier

PLAY_SPEED = 1


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
        self.date = None
        self.score = 0
        self.thinking_time = 0
        self.error_rate = 0.0
        self.instructions = instructions
        self.speed = 1
        self.rest_count = 0

    def _handle_state(self, state, syth):
        if state.get("stop") == 1:
            if self.rest_count < 3:
                self.rest(syth, 5)
                self.rest_count += 1
                return "retry"
            else:
                self.rest_count = 0
                return "exit"

        if state.get("emergency") == 1:
            self.finish_tesk(syth)
            return "exit"

        if state.get("dont_know") == 1:
            return "skip"

        return "ok"

    def _ask_until_done(self, question, syth, rcg, clf):
        while True:
            state, result = question.ask_question(syth, rcg, clf)

            action = self._handle_state(state, syth)
            if action == "retry":
                continue
            if action == "exit":
                return "exit", None
            if action == "skip":
                return "skip", None

            return "answered", result

    # The main cognitive rehabilitation task( Orientation task ) would be implemented in this function
    def perform_task(self, syth: speechSynthesize, rcg: speechRecognizer, clf: Classifier):

        print(self.instructions)
        syth.play_audio(
            text=self.instructions,
            filename=f"{self.name}_{self.id}_instruction.wav",
            playback_speed=1,
            is_synthesize=True
        )

        for idx, q in enumerate(self.list):
            if not isinstance(q, Question):
                raise ValueError("Non-Question object in task list")

            print(f"Main question {idx + 1}/{len(self.list)}")

            status, result = self._ask_until_done(q, syth, rcg, clf)
            if status == "exit":
                return 0
            if status == "skip":
                continue

            # 主问题答对
            if result == 1:
                continue

            # 主问题答错 → breakdown questions
            for bd in q.list:
                if not isinstance(bd, Question):
                    raise ValueError("Non-Question object in breakdown list")

                status, bd_result = self._ask_until_done(bd, syth, rcg, clf)
                if status == "exit":
                    return 0
                if status == "skip":
                    break
                if bd_result == 1:
                    break

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
                        is_synthesize=True)
        return 0

    def rest(self, syth: speechSynthesize, t: int):
        if self.rest_count < 3:
            duration = t * 60
            text = f"Let's have a {t} minutes rest."
            syth.play_audio(text=text,
                            filename=fr"rest_{t}.wav",
                            is_synthesize=True)
            time.sleep(duration)
        else:
            self.finish_task(syth)
            return 0

    def change_speed(self):
        return

    def logging(self):
        return 0


class Question(Item):
    def __init__(self, answer, q_id=0, name=None, question=None, is_synthesize=True):
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
        self.state = None

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


    def ask_question(self, syth: speechSynthesize, rcg: speechRecognizer, clf: Classifier):

        print(self.question)
        syth.play_audio(text=self.question,
                        filename=f"{self.name}_{self.id}_question.wav",
                        playback_speed=1
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

        if result == 1:
            self.score = 1
            syth.play_audio(text=self.correct_response,
                            filename=f"{self.name}_{self.id}_correct_response.wav",
                            playback_speed=1,
                            is_synthesize=self.is_synthesize)
            return state, result
        elif result == 0:
            if len(self.list) > 0:

                for i in self.list:
                    state, result = clf.state_match(input_text=text, target_text=i.answer)
                    if result == 0:
                        syth.play_audio(text=self.incorrect_response,
                                        filename=f"{self.name}_{self.id}_incorrect_response.wav",
                                        playback_speed=1,
                                        is_synthesize=self.is_synthesize)

                        return state, result

            syth.play_audio(text=self.correct_response, filename=f"{self.name}_{self.id}_correct_response.wav",
                            playback_speed=1)
            state["correct"] = 1
            result = 1

        return state, result

    def logging(self):
        return 0

def ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{ {1:'st', 2:'nd', 3:'rd'}.get(n % 10, 'th') }"

if __name__ == "__main__":
    # Initial question and task

    instruction = " my name is Emily. This task is a orientation task, try to answer the question after hearing the beep and avoid any guessing."

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

    print(f"{type(year)}...{year}...{type(month)}...{month}...{type(day_ordinal)}...{day_ordinal}...{type(weekday)}...{weekday}")

    p_0 = Patient(id=0, name="Jerry", address="Lennon studio")
    task_0 = Task(name="Orientation_task")
    task_0.set_instructions(instruction)
    q_main = Question(q_id=0, name="main_data", question=t_0, answer=a_0, is_synthesize=True)
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
