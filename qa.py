import nltk
nltk.download('punkt')
from nltk import sent_tokenize, word_tokenize
from multiprocessing import Pool
import csv
import json
import os
import torch
import numpy.random
import random
import demo
from transformers import pipeline
from tqdm import tqdm


fill_mask = pipeline(
    "fill-mask",
    model="google/electra-base-generator",
    tokenizer="google/electra-base-generator"
)

qa_pipeline = pipeline("question-answering",
                       model='deepset/roberta-base-squad2',
                       tokenizer='deepset/roberta-base-squad2')


def coreference(text):
    answers =  fill_mask(text)
    out = answers[0]['sequence']
    return " ".join(out.split()[1:len(out.split()) - 1])

def coreference_by_qa(text, story):
    return qa_pipeline(question=text, context=story, max_answer_len=5, handle_impossible_answer=False)['answer']


def convert_to_paracomet(sents):
  with open("example.jsonl", "w") as f:
    d= { "full_context" : sents, "id" : 1}
    json.dump(d, f)
def fetch_paracomet_results():
  data = None
  with open('/media/narrative_inference_demo/mem_beam_outputs.jsonl') as f:
    data = json.load(f)

  sent_keys = list(map(lambda i: "<|sent" + str(i) +"|>_generated_relations", range(5)))
  dims_keys = list(map(lambda i: "<|sent" + str(i) +"|>_generated_dims", range(5)))

  sents = list()
  dims = list()
  try:
    for i in range(5):
      sents.append(data[sent_keys[i]])
      dims.append(data[dims_keys[i]])
  except:
    pass
  return (sents, dims)


def execute_paracomet():
  demo.run_paracomet()

def generate_questions(attr_out, dims_out, count = 1):
  q_by_sent = list()
  coref_q_by_sent = list()
  clauses_by_sent = list()
  type_by_sent = list()

  for attr, dims in list(zip(attr_out, dims_out)):
    q_sent = list()
    coref_q = list()
    clauses_list = list()
    type_list = list()

    for clause_list, dim in list(zip(attr, dims)):
      #Take the first 30, shuffle, and then take the first 10 of those
      of_interest = clause_list[0:max(count, 10)]
      #print(len(of_interest))
      #print(clause_list)
      #random.shuffle(of_interest)
      for i in range(count):
        clause = of_interest[i]
        #These are all identical
        d = dim[0]

        #Only ones we care about
        if d == "<|xIntent|>":
          q_sent.append(f"What does {fill_mask.tokenizer.mask_token} do to need {clause}?")
          coref_q.append(f"Who needs {clause}?")
          clauses_list.append((d, clause))
          type_list.append(d)

        if d == "<|xNeed|>":
          q_sent.append(f"What does {fill_mask.tokenizer.mask_token} do {clause}?")
          coref_q.append(f"Who needs to {clause}?")
          clauses_list.append((d, clause))
          type_list.append(d)

        #if d == "<|xEffect|>":
        #  q_sent.append(f"What does {fill_mask.tokenizer.mask_token} do {clause}?")
        #  coref_q.append(f"Who needs to {clause}?")
        #  clauses_list.append((d, clause))
        #  type_list.append(d)

        #if d == "<|xAttr|>":
        #  q_sent.append(f"What did {fill_mask.tokenizer.mask_token} do to become {clause}?")
        #if d == "<|xWant|>":
        #  q_sent.append(f"Why does {fill_mask.tokenizer.mask_token} want " + clause + "?")
        #if d == "<|xReact|>":
        #  q_sent.append(f"Why does {fill_mask.tokenizer.mask_token} feel " + clause + "?")
    q_by_sent.append(q_sent)
    coref_q_by_sent.append(coref_q)
    clauses_by_sent.append(clauses_list)
    type_by_sent.append(type_list)
  return q_by_sent, coref_q_by_sent, clauses_by_sent, type_by_sent


scifi_stories_csv = None
with open("scifi_chunks_tgt_src_len5.csv") as f:
  scifi_stories_csv = list(csv.reader(f))

scifi_stories_csv = scifi_stories_csv[1:]
scifi_stories_csv = list(filter(lambda x: len(sent_tokenize(x[0])) != 1, scifi_stories_csv))


import warnings
warnings.filterwarnings("ignore")
story_prior = None

def determine_question(q, name):
  q_split = q.split()
  idx = q_split.index("[MASK]")
  if '\'s' == name[-2:]:
    name = name[:-2]
  if '.' == name[-1]:
    name = name[:-1]
  q_split[idx] = name

  return " ".join(q_split)


#Main code  
out_csv = list()
for story_idx, story in enumerate(tqdm_notebook(scifi_stories_csv[:20000])):
  #index sentence we start on for the source
  start_sent = len(sent_tokenize(story[0]))
  scifi_story = sent_tokenize(" ".join(story))
  convert_to_paracomet(scifi_story)

  if story_prior is None or story_prior != scifi_story:
    execute_paracomet()
    story_prior = scifi_story

  #Get questions
  out = fetch_paracomet_results()
  questions, coref_qs, clauses, types = generate_questions(*out, count=1)

  #print(coref_qs)
  #print(start_sent)

  #Set question index to the target/source split
  cur_coref = coref_qs[start_sent]
  cur_qs = questions[start_sent]
  q_out = list()

  for i in range(len(cur_coref)):
    name = coreference_by_qa(text = cur_coref[i], story=" ".join(scifi_story))
    q_out.append(determine_question(q=cur_qs[i], name=name))


  out_csv.append([story[0], story[1]].extend(q_out))

  if story_idx == 0:
    print(out_csv[-1])

  if len(out_csv) % 100 == 0 and len(out_csv) != 0:
    with open("scifi_dataset_qa.csv", "w+") as f:
      writer = csv.writer(f)
      writer.writerows(out_csv)
    print("Wrote step " + str(len(out_csv)) + " to file.")


