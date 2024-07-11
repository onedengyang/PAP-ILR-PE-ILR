import torch.nn as nn
import torch


class Ensemblemodel(nn.Module):
    def __init__(self, textEncoder,

                 text_hidden_size=None, num_class=None):

        super(Ensemblemodel, self).__init__()
        self.textEncoder = textEncoder
        self.text_hidden_size = text_hidden_size
        self.num_class = num_class
        for param in self.textEncoder.parameters():
            param.requires_grad = True

        self.fc = nn.Linear(text_hidden_size, int((text_hidden_size ) / 4))
        self.fc1 = nn.Linear(int((text_hidden_size ) / 4),
                             int((text_hidden_size ) / 8))
        self.fc2 = nn.Linear(int((text_hidden_size ) / 8), num_class)

    def forward(self, text_input_ids=None, code_input_ids=None, labels=None):
        text_output = self.textEncoder(text_input_ids, attention_mask=text_input_ids.ne(1))[1]


        # task1
        #combine_output = torch.cat([text_output, student_code_output], dim=-1)
        logits = self.fc(text_output)
        logits = self.fc1(logits)
        logits = self.fc2(logits)
        prob = torch.softmax(logits, -1)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
