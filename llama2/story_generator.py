import json

from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput, LLM
from vllm.lora.request import LoRARequest
from utils import *
import genre as import_genre
from genre import Genre
from ingest_story_json import ingest_story_json
import logging.config
from llm_engine import GenEngine

class StoryGenerator:
    def __init__(self, debug=False):
        self.structure_template = load_text_file('structure_template.txt')
        self.PLOT_POINTS_AMT = 9
        self.PLOT_POINTS_NUMBERING = [("1.1", "Exposition"),
                                      ("1.2", "Inciting Incident"),
                                      ("1.3", "Plot Point A"),
                                      ("2.1", "Rising Action"),
                                      ("2.2", "Midpoint"),
                                      ("2.3", "Plot Point B"),
                                      ("3.1", "Pre Climax"),
                                      ("3.2", "Climax"),
                                      ("3.3", "Denouement")]
        self.id = None
        self.title = None
        self.genre = None
        self.premise = None
        self.characters = None
        self.plot_points = None
        self.model = GenEngine()
        self.debug = debug
        # Load the logging configuration file
        logging.config.fileConfig('logging.conf')

        # Get a logger object
        self.logger = logging.getLogger()

    def ingest_structure(self, structure_path: str):
        json = ingest_story_json(structure_path)
        self.id = json['id']
        self.genre = import_genre.create(json['genre'])
        self.premise = json['story_generation']['premise']
        self.characters = json['story_generation']['characters']
        self.plot_points = json['story_generation']['plot_points']

    def _get_prompt_ready_premise(self):
        return "Premise: " + self.premise

    def _plot_points_to_prompt(self):
        prompt = """
[Story Events]
Setup

1.1 {}
1.2 {}
1.3 {}

Confrontation

2.1 {}
2.2 {}
2.3 {}

Resolution

3.1 {}
3.2 {}
3.3 {}
[/Story Events]
"""
        values = []

        for plot_point_idx, _ in self.PLOT_POINTS_NUMBERING:
            if plot_point_idx in self.plot_points:
                values.append(self.plot_points[plot_point_idx]['description'])
            else:
                values.append("")

        prompt = prompt.format(*values)
        return prompt

    def _construct_draft_prompt(self, plot_point_idx: str) -> str:
        # todo add characters to prompt
        prompt = """[INST] You are a renowned creative writer specialising in the genre of {}. {}{}
Using the provided context for the previous plot point, write a narrative section for the following plot point "{}". Be creative and ensure that the narrative sets up the upcoming passage. [/INST]"""

        pp, _ = zip(*self.PLOT_POINTS_NUMBERING)
        pp_true_index = pp.index(plot_point_idx)
        prev = pp_true_index > 0
        next = pp_true_index < len(pp) - 1

        prev = "\nIn the previous plot point:\n" + self.plot_points[
            pp[pp_true_index - 1]].get('summary', self._summarise_plot_point(
            pp[pp_true_index - 1])) + "\n" if prev else ""
        next = "\nIn the upcoming plot point:\n" + self.plot_points[
            pp[pp_true_index + 1]].get("description") + "\n" if next else ""

        prompt = prompt.format(self.genre.__str__(), prev, next,
                               self.plot_points[plot_point_idx]['description'])

        return prompt

    def _summarise_plot_point(self, plot_point_idx: str) -> str:
        self.logger.info(f"Summarising plot point {plot_point_idx}")
        sampling_params = SamplingParams(max_tokens=4096,
                                         # best_of=5,
                                         # set it same as max_seq_length in SFT Trainer
                                         temperature=0.3,
                                         # skip_special_tokens=True,
                                         # use_beam_search=True,
                                         )

        prompt = """[INST]You are a helpful assistant to a writer. You have been asked to summarise the following narrative section into a few sentences. The plot is as follows: {}[/INST]"""
        summary = ""
        try:
            text = self.plot_points[plot_point_idx]['text']
            prompt = prompt.format(text)
            output = self.model.process_requests(
                [(prompt, sampling_params, None)])
            self.plot_points[plot_point_idx]['summary'] = output[0].outputs[0].text
            summary = output[0].outputs[0].text
        except KeyError:
            summary = self.plot_points[plot_point_idx]['description']

        if self.debug:
            print(f"Summarised plot point {plot_point_idx} as\n-----------------------------\n{summary}")


        return summary

    def generate_plot_points(self):
        self.logger.info("Generating plot points")
        sampling_params = SamplingParams(max_tokens=4096,
                                         # best_of=5,
                                         # set it same as max_seq_length in SFT Trainer
                                         temperature=0.6,
                                         # skip_special_tokens=True,
                                         # use_beam_search=True,
                                         stop=[".\n"]
                                         )

        # generate the plot points
        for plot_point_idx, plot_point_desc in self.PLOT_POINTS_NUMBERING:
            if plot_point_idx in self.plot_points:
                continue
            else:
                # generate the plot point
                current_plot_prompt = self._plot_points_to_prompt()
                plot_point_prompt = """[INST]You are a renowned writer specialising in the genre of {}. You are able to create engaging narratives following a three act structure. Using the [Story Structure] outline, fill the [Story Events] suitable for the story as outlined in the premise.\n{}{}{}\nCreate a single event for the plot point {}, keep it concise and avoid repeating previous plot points.[/INST] {} {}:"""
                plot_point_prompt = plot_point_prompt.format(
                    self.genre.__str__(), self.structure_template,
                    self._get_prompt_ready_premise(), current_plot_prompt,
                    plot_point_idx, plot_point_idx, plot_point_desc)
                output = ""
                counter = 0
                while output == "":
                    counter += 1
                    output = self.model.process_requests(
                        [(plot_point_prompt, sampling_params, None)])
                    output = output[0].outputs[0].text
                self.logger.info(
                    f"Generated plot point {plot_point_idx} in {counter} attempts")
                # add the generated plot point to the plot points
                self.plot_points[plot_point_idx] = {
                    "description": output.strip()}
                if self.debug:
                    print(self._plot_points_to_prompt())

        # extract the characters and descriptions from the plot points
        # if self.characters is None:
        #     # extract named entities from the plot points
        #     EXTRACT_NAMED_ENTITIES = """[INST]You are an expert at extracting names from text.
        #     Using the following example: "Mario is a large man for brooklyn who meets the love of his life daniella" \n1. Mario\n2. Daniella \n
        #     Do the same for the following text: \n{}[/INST]"""
        #
        #     ne_prompts = EXTRACT_NAMED_ENTITIES.format("\n".join(
        #         [plot_point["description"] for plot_point in self.plot_points]))
        #
        #     outputs = llm.generate(ne_prompts, sampling_params)
        #
        #     for output in outputs:
        #         prompt = output.prompt
        #         generated_text = output.outputs[0].text
        #         print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        #
        #     ###todo turn the extracted named entities into characters

        self.logger.info("Finished generating plot points")
        return

    def generate_story(self, t=0.7):
        self.logger.info("Generating story")
        sampling_params = SamplingParams(max_tokens=4096,
                                         temperature=t,
                                         top_k=50,
                                         repetition_penalty=0.5,
                                         frequency_penalty=1.19,
                                         )

        for plot_point_idx, plot_point_desc in self.PLOT_POINTS_NUMBERING:
            self.logger.info(f"Generating plot point {plot_point_idx}")
            draft_prompt = self._construct_draft_prompt(plot_point_idx)
            self.plot_points[plot_point_idx]['draft_prompt'] = draft_prompt
            output = self.model.process_requests(
                [(draft_prompt, sampling_params, self.genre)])
            output = output[0].outputs[0].text

            self.plot_points[plot_point_idx]['text'] = output
            self.logger.info(f"Generated plot point {plot_point_idx}")

            if self.debug:
                print(f"Generated plot point {plot_point_idx} as\n-----------------------------\n{output}")

        self.logger.info("Finished generating story")

        #create a text file with the story
        with open(f'outputs/{self.id}.txt', 'w') as f:
            for plot_point_idx, plot_point_desc in self.PLOT_POINTS_NUMBERING:
                f.write(f"{plot_point_idx}: {plot_point_desc}\n")
                f.write(f"{self.plot_points[plot_point_idx]['text']}\n\n")

        #create a json file with the plot points
        with open(f'outputs/{self.id}.json', 'w') as f:
            json.dump(self.plot_points, f)


sg = StoryGenerator(debug=True)
sg.ingest_structure('example.json')
sg.generate_plot_points()
sg.generate_story()
