# Automating OS Upcoding in CHIS Data

## Introduction

The California Health Interview Survey (CHIS) is a leading source of credible and comprehensive data on the health and healtcare needs of California’s large and 
diverse population. Among the various type of questions included in the survey, one is structured with an additional 
component called an OS (Other Specify) response. This is an open text category included to allow respondents to include a short response if they feel the other 
responses do not cover their response. An example of such a question is: 

**‘QA19_C15’ [AC83C]** - What best describes your reasons for using 
e-cigarettes?
*Check all that apply*
- 1 To quit smoking
- 2 To replace smoking
- 3 To cut down or reduce smoking
- 4 To use in places where smoking is not allowed
- 5 To just try it out of curiosity
- 6 To avoid the lingering odor of cigarettes
- 7 To help me concentrate/ stay alert
- 8 Because they come in many flavors
- 9 Because they are less expensive than cigarettes
- 10 Because they are healthier than cigarettes
- 11 For enjoyment or social reasons
- 12 To reduce stress, anxiety, or pain
- 91 Other (Specify: ____________).

One way to facilitate analysis is to attempt to either include these OS responses in categories included in the question, or inlcude them in newly created categories 
indentified from trends seen in the text data. Usually this task requires large amounts of time for human review, as well as extensive training in the task. 
Therefore, it is of interest to find a method by which this process can be automated. The task basically amounts to a text classification problem. Machine learning 
has demonstrated good performance on text classification tasks, which naturally makes the method a candidate. In particular, Large Language Models 
(LLM) have exploded in popularity and have shown exceptionally good performance in working with text data. In this repo, we explore their usage in automating the 
process of OS reponse classification (OS upcoding). 

## Methods

coming soon...

