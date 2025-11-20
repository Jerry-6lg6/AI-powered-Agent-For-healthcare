import time
from abc import ABC, abstractmethod
from collections import defaultdict
import datetime
from speech_synthesis import speechSynthesize
from speech_Recognition import speechRecognizer
from classifier import Classifier

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

    # The main cognitive rehabilitation task( Orientation task ) would be implemented in this function
    def perform_task(self, syth: speechSynthesize, rcg: speechRecognizer, clf: Classifier):
        """

        :param syth: speech Synthesizer
        :param rcg: speech recognizer
        :param clf: Classifier
        :return:
        """
        print(self.instructions)
        syth.play_audio(text=self.instructions,
                        filename=f"{self.name}_{self.id}_instruction.wav",
                        playback_speed=1,
                        is_synthesize=False)
        for i in self.list:
            if type(i) is not Question:
                raise ValueError
            else:
                # Asking main question
                result = i.ask_question(syth, rcg, clf)
                # Correct-> end
                if result == 0:
                    continue
                # Incorrect-> Asking break-down questions
                else:
                    for j in i.list:
                        if type(i) is not Question:
                            raise ValueError
                        j.ask_question(syth, rcg, clf)

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

    def calculate(self):
        if len(self.list) == 0:
            return 0
        else:
            for i in self.list:
                self.score += i.score
            self.error_rate = self.score / len(list)
            return 0

    def set_avg_thinking_time(self, t_time:float):
        if len(self.list) == 0:
            self.thinking_time = t_time
        else:
            total_time = 0.0
            for i in self.list:
                total_time += i.thingking_time
            self.avg_thinking_time = total_time / len(list)

    def ask_question(self, syth:speechSynthesize, rcg:speechRecognizer, clf:Classifier):

        print(self.question)
        syth.play_audio(text=self.question,
                        filename=f"{self.name}_{self.id}_question.wav",
                        playback_speed=1
                        )

        time.sleep(1)

        text, time_0, time_1 = rcg.record_audio()

        self.thinking_time = time_0
        result = clf.simplematch(input_text=text, target_text=self.answer)
        print(f"the result is {result}, input_text is P{text}, target test is {self.answer}")
        if result == 0:
            self.score = 1
            syth.play_audio(text=self.correct_response,
                            filename=f"{self.name}_{self.id}_correct_response.wav",
                            playback_speed=1,
                            is_synthesize=self.is_synthesize)
            return 0
        elif result == 1:
            if len(self.list)>0:

                for i in self.list:
                    r = clf.simplematch(input_text=text, target_text=i.answer)
                    if r == 1:
                        syth.play_audio(text=self.incorrect_response,
                                        filename=f"{self.name}_{self.id}_incorrect_response.wav",
                                        playback_speed=1,
                                        is_synthesize=self.is_synthesize)

                        return 1

            syth.play_audio(text=self.correct_response, filename=f"{self.name}_{self.id}_correct_response.wav",
                                playback_speed=1)

        return 0

    def logging(self):
        return 0


if __name__ == "__main__":
    # Initial question and task

    instruction = "This task is a orientation task, try to answer the question after hearing the beep and avoid any guessing."

    t_0 = "What specific date of today"
    t_00 = "what year is it this year? "
    t_01 = "What mouth is it this mouth? "
    t_02 = "What data is it today"
    t_03 = "What day is it today"

    today = datetime.date.today()
    year = today.strftime("%Y")
    month = today.strftime("%B")
    day = today.strftime("%d")
    weekday = today.strftime("%A")

    print(f"{type(year)}_({year})")

    a_0 = f"{weekday}, {month} the {day}, {year}"
    a_00 = year
    a_01 = month
    a_02 = day
    a_03 = weekday

    print(f"{type(year)}...{year}...{type(month)}...{month}...{type(day)}...{day}...{type(weekday)}...{weekday}")

    p_0 = Patient(id=0, name="Jerry", address="Lennon studio")
    task_0 = Task(name="Orientation_task")
    task_0.set_instructions(instruction)
    q_main = Question(q_id=0, name="main_data", question=t_0, answer=a_0)
    q_0 = Question(q_id=1, name="bd_0", question=t_00, answer=a_00)
    q_1 = Question(q_id=2, name="bd_1", question=t_01, answer=a_01)
    q_2 = Question(q_id=3, name="bd_3", question=t_02, answer=a_02)
    q_3 = Question(q_id=4, name="bd_4", question=t_03, answer=a_03)

    # list_bd = [q_0, q_1, q_2, q_3]
    list_bd = [q_3]

    q_main.set_list(list_bd)
    list_q = [q_main]
    task_0.set_list(list_q)

    sr = speechRecognizer(model_name="faster_whisper")
    tts = speechSynthesize(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    classifier_0 = Classifier()

    task_0.perform_task(tts, sr, classifier_0)



