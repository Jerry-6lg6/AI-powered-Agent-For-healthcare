import time
from abc import ABC, abstractmethod
from collections import defaultdict
import datetime
from database import PatientDatabase

import torch

from speech_synthesis import speechSynthesize
from speech_Recognition import speechRecognizer
from classifier import Classifier

PLAY_SPEED = 1
WAIT_TIME = 1
SPEED = 1
SPEED_SLOW = 0.85
REST_TIME = 3
CHANGEABLE = True
UNCHANGEABLE = False
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
            raise ValueError("input value should be a list and should not be an empty list.")
        return 0

    def get_list(self):
        return self.list

    def set_dic(self, key, value):
        self._dic[key] = value
        return 0

    def get_dic(self):
        return self._dic

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
        self.speed = SPEED
        self.rest_count = 0
        self.retry_count = 0
        self.time = time
        # Patient attribute — long term solution so task always knows its patient
        self.patient = None
        self.session_id = None

    def _handle_state(self, state, syth, current_state: TaskState = None,
                      session_record=None, db=None):
        # Fix 3: session_record and db passed in as parameters
        # so this method can safely read and write to them

        if state.get("stop") == 1:
            if self.rest_count < 3:
                self.rest(syth, 1)
                self.rest_count += 1
                # Update session record and flag event if available
                if session_record is not None:
                    session_record["rest_breaks_taken"] += 1
                if db is not None:
                    db.flag_event(self.patient.id, self.session_id,
                                  "rest_requested", {"rest_count": self.rest_count})
                return "return"
            else:
                self.rest_count = 0
                return "exit"

        elif state.get("emergency") == 1:
            self.finish_task(syth)
            if db is not None:
                db.flag_event(self.patient.id, self.session_id, "emergency", {})
            return "exit"

        elif state.get("dont_know") == 1:
            self.speed = SPEED_SLOW
            # Update session record and flag event if available
            if session_record is not None:
                session_record["speed_changes"].append({
                    "old_speed": 1.0,
                    "new_speed": SPEED_SLOW,
                    "reason": "dont_know"
                })
            if db is not None:
                db.flag_event(self.patient.id, self.session_id,
                              "speed_changed",
                              {"old_speed": 1.0,
                               "new_speed": SPEED_SLOW,
                               "reason": "dont_know"})
            print(f"NEW SPEED : {self.speed}")
            decision = self.encourage(syth)
            return decision

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

    def perform_task(self, syth: speechSynthesize, rcg: speechRecognizer,
                     clf: Classifier, db: PatientDatabase):

        # ── TOP SECTION ───────────────────────────────────────────────
        print(self.instructions)
        state = TaskState.INIT
        q_index = 0
        bd_index = 0
        current_question = None
        breakdown_list = []
        pre_state = None
        retry_count = 0
        turn_record = None
        result = None

        # Generate unique session ID
        self.session_id = f"{self.name}_{self.patient.id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # READ — load patient config from database
        config = db.load_session_config(self.patient.id)

        # Apply config to agent behaviour
        self.speed = config.recommended_speed
        self.wait_time = config.preferred_wait_time
        self.bd_retry_budget = config.bd_retry_budget

        # Initialise session record
        session_record = {
            "session_id"        : self.session_id,
            "patient_id"        : self.patient.id,
            "task_name"         : self.name,
            "date"              : str(datetime.datetime.now().date()),
            "time_of_day"       : "morning" if datetime.datetime.now().hour < 12
                                  else "afternoon",
            "final_state"       : None,
            "total_score"       : 0,
            "session_duration"  : None,
            "rest_breaks_taken" : 0,
            "speed_changes"     : [],
            "reset_required"    : False,
            "questions"         : []
        }

        # Record session start time for duration calculation at the end
        session_start = datetime.datetime.now()

        # ── MIDDLE SECTION ────────────────────────────────────────────
        while state not in (TaskState.FINISH, TaskState.EXIT):

            # INIT
            if state == TaskState.INIT:
                if len(self.list) == 0:
                    state = TaskState.FINISH
                    continue
                q_index = 0
                intro = "Hi, my name is Jennet, your personal medical assistant."
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

            # READY CHECK
            elif state == TaskState.READY_CHECK:
                check_result = self.ready_check(syth, rcg, clf)

                if check_result == "ready":
                    now = time.localtime()
                    formatted_time = time.strftime("It is %I:%M %p on %A, %B %d, %Y.", now)
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

            # WAITING FOR KEYWORD
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
                    syth.play_audio(
                        text=self.instructions,
                        filename=f"{self.name}_{self.id}_{self.speed}_instruction.wav",
                        playback_speed=self.speed,
                        is_synthesize=UNCHANGEABLE
                    )
                    state = TaskState.INTERACT_MAIN

                elif keyword_result == "timeout":
                    timeout_text = "Let me check if you're ready again."
                    syth.play_audio(
                        text=timeout_text,
                        filename=f"keyword_timeout_{self.speed}.wav",
                        playback_speed=self.speed,
                        is_synthesize=UNCHANGEABLE
                    )
                    state = TaskState.READY_CHECK

            # INTERACT MAIN
            elif state == TaskState.INTERACT_MAIN:
                current_question = self.list[q_index]

                # Fix 5: unpack 3 values
                q_state, result, turn_record = current_question.ask_question(
                    syth, rcg, clf, speed=self.speed)

                # Fix 3: pass session_record and db into _handle_state
                action = self._handle_state(q_state, syth, state,
                                            session_record=session_record, db=db)
                print(f"The state of main question is {q_state}\n"
                      f"The result is {result}\nThe action is {action}")

                # Fill in action_taken on the turn record
                if turn_record is not None:
                    turn_record["action_taken"] = action

                if action == "exit":
                    state = TaskState.EXIT
                elif action == "retry_dont_know":
                    continue
                elif action == "return":
                    text = "Let's go back to our task."
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
                    state = TaskState.NEXT
                else:  # wrong — enter breakdown
                    breakdown_list = current_question.list
                    bd_index = 0
                    if len(breakdown_list) == 0:
                        state = TaskState.NEXT
                    else:
                        text_bd = "Let's break it down."
                        syth.play_audio(text=text_bd,
                                        filename=f"breakdown_bridge_{self.speed}.wav",
                                        playback_speed=self.speed,
                                        is_synthesize=UNCHANGEABLE)
                        bd_retry_used = False
                        state = TaskState.INTERACT_BREAKDOWN

            # INTERACT BREAKDOWN
            elif state == TaskState.INTERACT_BREAKDOWN:

                bd_question = breakdown_list[bd_index]

                # Correctly asks the breakdown question (not current_question)
                q_state, result, turn_record = bd_question.ask_question(
                    syth, rcg, clf, speed=self.speed)

                # Fix 3: pass session_record and db into _handle_state
                decision = self._handle_state(q_state, syth, state,
                                              session_record=session_record, db=db)

                # Fill in action_taken on the turn record
                if turn_record is not None:
                    turn_record["action_taken"] = decision

                if decision == "exit":
                    state = TaskState.EXIT

                elif decision == "return":
                    text = "Let's go back to our task."
                    syth.play_audio(text=text,
                                    filename=f"return_{self.speed}.wav",
                                    playback_speed=self.speed,
                                    is_synthesize=UNCHANGEABLE)

                elif decision == "retry":
                    if not bd_retry_used:
                        bd_retry_used = True
                        # Flag the retry event
                        db.flag_event(self.patient.id, self.session_id,
                                      "bd_retry_used",
                                      {"question_id": bd_question.id,
                                       "result_after_retry": result})
                        text = "Let's try that once more."
                        syth.play_audio(text=text,
                                        filename=f"bd_retry_once_{self.speed}.wav",
                                        playback_speed=self.speed,
                                        is_synthesize=UNCHANGEABLE)
                        continue
                    else:
                        bd_retry_used = False
                        bd_index += 1
                        if bd_index >= len(breakdown_list):
                            state = TaskState.NEXT
                        else:
                            state = TaskState.INTERACT_BREAKDOWN
                            text = "Let's move on."
                            syth.play_audio(text=text,
                                            filename=f"bd_move_on_{self.speed}.wav",
                                            playback_speed=self.speed,
                                            is_synthesize=UNCHANGEABLE)
                else:  # still wrong
                    bd_retry_used = False
                    bd_index += 1
                    retry_count = 0
                    if bd_index >= len(breakdown_list):
                        state = TaskState.NEXT
                    else:
                        state = TaskState.INTERACT_BREAKDOWN

            # NEXT
            elif state == TaskState.NEXT:

                # Build and append the question record
                question_record = {
                    "question_id"             : current_question.id,
                    "question_text"           : current_question.question,
                    "expected_answer"         : current_question.answer,
                    "escalated_to_breakdown"  : (bd_index > 0),
                    "final_result"            : result,
                    "which_sub_question_failed": [],
                    "turns"                   : [turn_record] if turn_record else []
                }
                session_record["questions"].append(question_record)
                session_record["total_score"] += current_question.score

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

        # ── BOTTOM SECTION ────────────────────────────────────────────
        # Fix 6: finalise and save BEFORE finish_task()
        # Fix 4: correct key name "final_state"
        session_record["final_state"] = (
            "FINISH" if state == TaskState.FINISH else "EXIT"
        )
        session_record["session_duration"] = (
            datetime.datetime.now() - session_start
        ).total_seconds()

        # WRITE — save session and update patient profile
        db.save_session(self.patient.id, session_record)
        db.close()

        # Fix 1: only one finish block, after the save
        # Fix typo: torch.cuda (not torch.cude)
        if state == TaskState.FINISH:
            self.finish_task(syth)
        elif state == TaskState.EXIT:
            print("Emergency, we need some help here.")
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
                total_time += i.thinking_time
            self.thinking_time = total_time / len(list)

    def finish_task(self, syth: speechSynthesize):
        text = "That is the end of the task. See you."
        syth.play_audio(text=text,
                        filename=fr"break_{self.speed}.wav",
                        is_synthesize=UNCHANGEABLE,
                        playback_speed=self.speed)
        return 0

    def encourage(self, syth: speechSynthesize):
        if self.retry_count < 1:
            text = "Don't worry, I would slow down a little bit. Let's have another try."
            syth.play_audio(text=text,
                            filename=fr"encourage_{self.speed}.wav",
                            is_synthesize=UNCHANGEABLE,
                            playback_speed=self.speed)
            return "retry_dont_know"
        else:
            self.retry_count = 0
            text = "What about the next one."
            syth.play_audio(text=text,
                            filename=fr"next_{self.speed}.wav",
                            is_synthesize=UNCHANGEABLE,
                            playback_speed=self.speed)
            return "continue"

    def rest(self, syth: speechSynthesize, t: int):
        if self.rest_count < 3:
            duration = t * 60
            text_0 = f"Let's have a {t} minute rest."
            syth.play_audio(text=text_0,
                            filename=fr"rest_{t}_{self.speed}.wav",
                            is_synthesize=CHANGEABLE,
                            playback_speed=self.speed)
            time.sleep(duration)
            text_1 = "Let's go back to the task."
            syth.play_audio(text=text_1,
                            filename=fr"rest_return_{t}_{self.speed}.wav",
                            is_synthesize=UNCHANGEABLE,
                            playback_speed=self.speed)
        else:
            self.finish_task(syth)
            return 0

    def ready_check(self, syth: speechSynthesize, rcg: speechRecognizer,
                    clf: Classifier):
        """
        Check if patient is ready to start.
        Returns: 'ready', 'waiting', or 'exit'
        """
        text = "If you are ready to start this task, please say yes after hearing the beep."
        syth.play_audio(
            text=text,
            filename=f"ready_check_{self.speed}.wav",
            playback_speed=self.speed,
            is_synthesize=UNCHANGEABLE
        )
        time.sleep(1)
        response, _, _ = rcg.record_audio(duration=5)

        state, score = clf.state_match(input_text=response, target_text="yes")

        if state.get("emergency") == 1:
            return "exit"

        if state.get("correct") == 1:
            confirm_text = "Great, let's begin."
            syth.play_audio(
                text=confirm_text,
                filename=f"ready_confirmed_{self.speed}.wav",
                playback_speed=self.speed,
                is_synthesize=UNCHANGEABLE
            )
            return "ready"
        else:
            wait_text = f"No problem. I'll wait for {WAIT_TIME} minute."
            syth.play_audio(
                text=wait_text,
                filename=f"waiting_for_keyword_{self.speed}.wav",
                playback_speed=self.speed,
                is_synthesize=CHANGEABLE
            )
            return "waiting"

    def wait_for_keyword(self, rcg: speechRecognizer, keyword="jennet",
                         timeout=300):
        """
        Continuously monitor for keyword using VAD.
        Returns: 'detected' or 'timeout'
        """
        start_time = time.time()
        check_interval = 2
        print(f"Sleeping for {timeout}s...")
        time.sleep(timeout)
        return "timeout"

    def change_speed(self):
        return

    def logging(self):
        return 0


class Question(Item):
    def __init__(self, answer, q_id=0, name=None, question=None,
                 is_synthesize=True, end_text=None):
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
        self.correct_response = f"Great, the answer is {self.answer}."
        self.incorrect_response = f"Don't worry, the answer is {self.answer}."
        self.is_synthesize = is_synthesize
        self.state_list = []
        self.end_text = end_text
        self.speaking_time = 0

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
                total_time += i.thinking_time
            self.avg_thinking_time = total_time / len(list)

    def ask_question(self, syth: speechSynthesize, rcg: speechRecognizer,
                     clf: Classifier, speed=1):

        print(self.question)
        syth.play_audio(text=self.question,
                        filename=f"{self.name}_{self.id}_{speed}_question.wav",
                        playback_speed=speed)

        time.sleep(1)
        text, time_0, time_1 = rcg.record_audio()

        self.thinking_time = time_0
        self.speaking_time = time_1 - time_0
        state, score = clf.state_match(input_text=text, target_text=self.answer)
        result = state["correct"]

        # Fix 7: build turn_record FIRST before any returns
        # so every return path includes it
        turn_record = {
            "thinking_time"           : self.thinking_time,
            "speaking_time"           : self.speaking_time,
            "classifier_score"        : score,
            "special_state_triggered" : (score == 2),
            "correct"                 : result,
            "action_taken"            : None,
            "bd_retry_used"           : False,
            "bd_retry_successful"     : False
        }

        print(f"Result: {result} | Input: {text} | Target: {self.answer}")

        # Special state — return all three
        if score == 2:
            print("Special state detected...")
            return state, result, turn_record

        # Correct answer — return all three
        if result == 1:
            self.score = 1
            syth.play_audio(text=self.correct_response,
                            filename=f"{self.name}_{self.id}_{speed}_correct_response.wav",
                            playback_speed=speed,
                            is_synthesize=CHANGEABLE)
            self.state_list.append(state)
            return state, result, turn_record

        # Check if all sub-question answers were present in the response
        if len(self.list) > 0:
            matched = 0
            for sub_q in self.list:
                sub_state, _ = clf.state_match(input_text=text,
                                               target_text=sub_q.answer)
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
                return final_state, 1, turn_record

        # Incorrect — return all three
        syth.play_audio(
            text=self.incorrect_response,
            filename=f"{self.name}_{self.id}_{speed}_incorrect_response.wav",
            playback_speed=speed,
            is_synthesize=CHANGEABLE
        )
        print(f"State: {state}")
        self.state_list.append(state)
        return state, result, turn_record

    def logging(self):
        return 0


def ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{  {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


if __name__ == "__main__":

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Define question texts
    t_0  = "what the date is today"
    t_00 = "what the year is"
    t_01 = "what the month is"
    t_02 = "what the date of the month is"
    t_03 = "what day of the week it is"

    # Calculate today's date components
    today      = datetime.date.today()
    year       = today.strftime("%Y")
    month      = today.strftime("%B")
    day_int    = today.day
    day_ordinal = ordinal(day_int)
    weekday    = today.strftime("%A")

    # Define answers
    a_0  = f"{weekday}, {month} the {day_ordinal}, {year}"
    a_00 = year
    a_01 = month
    a_02 = day_ordinal
    a_03 = weekday

    now = time.localtime()
    formatted_time = time.strftime("It is %I:%M %p on %A, %B %d, %Y.", now)
    instruction = ("There is an orientation task in your planner. "
                   "Try to answer the question after hearing the beep.")

    end_text = (f"Today is {a_0}. If you need to check what the date is, "
                f"it is written on your Orientation Chart or you can ask "
                f"any member of staff.")

    # Create patient and task
    p_0    = Patient(id=0, name="Jerry", address="Lennon Studio")
    task_0 = Task(name="Orientation_task")
    task_0.set_instructions(instruction)

    # Create questions
    q_main = Question(q_id=0, name="main_data", question=t_0,
                      answer=a_0, is_synthesize=True, end_text=end_text)
    q_0    = Question(q_id=1, name="bd_0", question=t_00,
                      answer=a_00, is_synthesize=True)
    q_1    = Question(q_id=2, name="bd_1", question=t_01,
                      answer=a_01, is_synthesize=True)
    q_2    = Question(q_id=3, name="bd_2", question=t_02,
                      answer=a_02, is_synthesize=True)
    q_3    = Question(q_id=4, name="bd_3", question=t_03,
                      answer=a_03, is_synthesize=True)

    # Attach breakdown questions to main question
    list_bd = [q_0, q_1, q_2]
    q_main.set_list(list_bd)
    list_q = [q_main]
    task_0.set_list(list_q)

    # Initialise speech and classifier
    sr          = speechRecognizer(model_name="faster_whisper", device="cpu")
    tts         = speechSynthesize(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    classifier_0 = Classifier()

    # Initialise database
    db = PatientDatabase("patients.db")
    db.initialise()
    db.register_patient(p_0)

    # Attach patient to task and run
    task_0.patient = p_0
    task_0.perform_task(tts, sr, classifier_0, db)
