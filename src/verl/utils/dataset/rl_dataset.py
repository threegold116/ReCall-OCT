# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import List, Union, Optional
import copy
import datasets
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from omegaconf import ListConfig, DictConfig

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.dataset.template import prompt_template_dict
import random

def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: dict, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.max_prompt_length = config.get("max_prompt_length", 1024)

        self.return_raw_chat = config.get('return_raw_chat', False)
        self.truncation = config.get('truncation', 'error')
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.use_re_call = config.get('use_re_call', True)
        self.prompt_template_name = config.get('prompt_template_name', 're_call_template_sys')
        self.prompt_template = prompt_template_dict[self.prompt_template_name] if self.prompt_template_name else None

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local
        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _pack_re_call_input(self, prompt_template, func_schemas, user_input):
        return self.tokenizer.apply_chat_template([
                {'role': 'system', 'content': prompt_template.format(func_schemas=func_schemas)}, 
                {'role': 'user', 'content': user_input}
            ], add_generation_prompt=True, tokenize=False) + "<think>"
    #THREEGOLDCHANGE:增加budget
    def _pack_re_call_budget_input(self, prompt_template, func_schemas, user_input):
        cost_sentence_template = "{function_name}: {cost}"
        func_schemas_list = eval(func_schemas.replace("null","None").replace("true","True").replace("false","False"))
        cost_dict = {}
        cost_sentences = []
        for idx in range(len(func_schemas_list)):
            cost = random.randint(1, 10)
            try:
                function_name = func_schemas_list[idx]['function']['name']
            except:
                function_name = func_schemas_list[idx]['name']
            cost_dict[function_name] = cost
            cost_sentences.append(cost_sentence_template.format(function_name=function_name, cost=cost))
        cost_sentence = "\n".join(cost_sentences)
        if cost_dict == {}:
            cost_dict = None
            pass
        # cost_sentence = cost_sentence_template.format(function_name=func_schemas_list[0]['function']['name'], cost=cost)
        # for idx in range(1,len(func_schemas_list)-1):
        #     cost_sentence += ", "
        #     cost = random.randint(1, 10)
        #     cost_sentence += cost_sentence_template.format(function_name=func_schemas_list[idx]['function']['name'], cost=cost)
        # if len(func_schemas_list)>1:
        #     cost = random.randint(1, 10)
        #     cost_sentence +=" and " + cost_sentence_template.format(function_name=func_schemas_list[-1]['function']['name'], cost=cost)
        return self.tokenizer.apply_chat_template([
                {'role': 'system', 'content': prompt_template.format(func_schemas=func_schemas,cost_sentence=cost_sentence)}, 
                {'role': 'user', 'content': user_input}
            ], add_generation_prompt=True, tokenize=False) + "<think>",cost_dict
    #THREEGOLDCHANGE:增加budget
    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f'dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            if self.use_re_call:
                if self.prompt_template_name == 're_call_template_sys':
                    self.dataframe = self.dataframe.filter(
                        lambda doc: len(tokenizer.encode(self._pack_re_call_input(self.prompt_template, doc['extra_info']['func_schemas'], doc[prompt_key]))
                                    ) <= self.max_prompt_length,
                        num_proc=self.num_workers,
                        desc=f"Filtering prompts longer than {self.max_prompt_length} tokens")
                elif self.prompt_template_name == 're_call_template_budget_sys':
                    self.dataframe = self.dataframe.filter(
                        lambda doc: len(tokenizer.encode(self._pack_re_call_budget_input(self.prompt_template, doc['extra_info']['func_schemas'], doc[prompt_key])[0])
                                    ) <= self.max_prompt_length,
                        num_proc=self.num_workers,
                        desc=f"Filtering prompts longer than {self.max_prompt_length} tokens")
            else:
                self.dataframe = self.dataframe.filter(
                    lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)
                                ) <= self.max_prompt_length,
                    num_proc=self.num_workers,
                    desc=f"Filtering prompts longer than {self.max_prompt_length} tokens")

            print(f'filter dataset len: {len(self.dataframe)}')

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_data_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]

        chat = row_dict.pop(self.prompt_key)
        # THREEGOLDCHANGE:增加budget
        cost_dict = None
        # THREEGOLDCHANGE
        if self.use_re_call:
            if self.prompt_template_name == 're_call_template_sys':
                prompt_with_chat_template = self._pack_re_call_input(self.prompt_template, row_dict['extra_info']['func_schemas'], chat)
            elif self.prompt_template_name == 're_call_template_budget_sys':
                prompt_with_chat_template,cost_dict = self._pack_re_call_budget_input(self.prompt_template, row_dict['extra_info']['func_schemas'], chat)
            elif self.prompt_template_name == 're_call_template_times_sys':
                prompt_with_chat_template = self._pack_re_call_input(self.prompt_template, row_dict['extra_info']['func_schemas'], chat)
        else:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        is_multi_modal = self.image_key in row_dict
        if is_multi_modal:  # expand image token
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(image) for image in row_dict.pop(self.image_key)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                              self.processor.image_token)
        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        if is_multi_modal:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        # handle ground truth is list 
        if isinstance(row_dict['reward_model']['ground_truth'], np.ndarray):
            row_dict['reward_model']['ground_truth'] = row_dict['reward_model']['ground_truth'].tolist()

        # TODO: not use hard code here, use a placeholder map to replace such url adaptively
        if self.use_re_call:
            if row_dict['data_source'] == 'musique_re_call':
                assert self.config.get('search_url', None) is not None, "search_url for musique_re_call dataset is not set"
                row_dict['env'] = row_dict['extra_info']['env'].replace('<search-url-placeholder>', self.config.get('search_url', ''))
            else:
                row_dict['env'] = row_dict['extra_info']['env']
            if cost_dict is not None:
                row_dict["cost_dict"] = cost_dict
                # for function_name,cost in cost_dict.items():
                #     [f"{function_name}_cost"] = cost

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()
