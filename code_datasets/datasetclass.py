import pandas as pd
import numpy as np
import os
import random
import torch
import torch.nn as nn

class CodeContestDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data):
        super().__init__()
        self.raw_data = raw_data
    def __len__(self):
        return self.raw_data.shape[0]
    def __getitem__(self, idx):
        row = self.raw_data.loc[idx]
        task_id = row['task_id']
        label = row['solution'].replace("\\n", "\n")
        input_description = row['prompt'].replace("\\n", "\n")
        return  task_id, input_description, label