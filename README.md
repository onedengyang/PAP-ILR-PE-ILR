# PAP-ILR-PE-ILR
Code for paper "When Large Language Models Meet Issue-commit Link Recovery: PAP-ILR or PE-ILR?". The remaining parts will be published if the paper is accepted.


##  GPTLink - Issue-Commit Link Recovery Tool

GPTLink is a tool that leverages GPT models for issue-commit link recovery(ILR) and link enhancement. It uses various prompts to improve the accuracy of links, helping development teams manage and track changes in their projects more efficiently.


### Features
1. Zero-shot ILR with Prompt Engineering. The proposed Prompt Engineering for ILR (PE-ILR) achieves zero-shot ILR, requiring no training data and directly guiding large models to perform ILR tasks using various prompt templates. The results indicate that PE-ILR with GPT-4 outperforms state-of-the-art methods

2. Link Augmentation by GPT. The proposed link augmentation with ChatGPT imitates the problem of limited true linksfrom the semantic level and data imbalance. The experimental results confirm that ChatGPT can generate high-quality true links. It also demonstrate that increasing the number of samples improves the performance of the pre-trained model.

3. Real-time ILR.

### Usage
**Input Data**
Prepare your input data, including a list of issues and commits from your repository. The issues should contain relevant information such as “Issue_KEY” and “Issue_Text”. Commits should include “,Commit_SHA” and “Commit_Code”.

Here is an example of how your input data files might look:



**Path:** F:\gpt_link\logging-log4net_TEST.csv
|Issue_KEY|Commit_SHA|Issue_Text|Commit_Text|Commit_Code|
|  ----  | ----  |----  |----  |----  |
| LOG4NET-354  |12a0c7397eb19eafbbcdfa0a93b1ae03740ad212... | E-mail encoding configuration setting for SmtpAppender.... | LOG4NET-354 reverted changes from revision 1489736 since we do not want to change the public API... | MODIFY LogImpl.cs MODIFY ILog.cs |
|  .....  | .....  .....  |..... |.....  |
| LOG4NET-215  |451cce90f726c70f60f81848ff75413efb4e3e34... | Exception on Convert for return %class{1} name in function.... | fix bounds-checks in NamedPatternConverter.  LOG4NET-215| MODIFY NamedPatternConverter.....|



**Demo**
For a detailed walkthrough on how to use GPTLink, watch our video tutorial:

![GPTLINK](/GPTLINK.gif)









