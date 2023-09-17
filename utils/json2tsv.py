import io
import os
import json
from tqdm import tqdm
from argparse import ArgumentParser


def num_there(s):
    return all(i.isdigit() for i in s)

def str_there(s):
    return all(i.isalpha() for i in s)

parser = ArgumentParser()
parser.add_argument('-i', '--input_dir', default='pred')
parser.add_argument('-o', '--out_dir', default='pred', help='Path to the output directory')
args = parser.parse_args()

input_dir = args.input_dir
out_dir = args.out_dir

f_name = input_dir.split('/')[-1]
f_fileds = f_name.split('.')
# out_path = os.path.join(out_dir, '_'.join(f_fileds[:-1]))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
# print(f_name)


docs = {}
with io.open(input_dir, encoding='utf-8') as f:
    lines = f.read().strip().split('\n')

    for example_num, line in tqdm(enumerate(lines), desc=f_name):
        example = json.loads(line)
        docname = '_'.join(example['doc_key'].split('/'))

        # if example['doc_key'] != 'bc/phoenix/00/phoenix_0000_5':
        #     continue

        count = 0
        merged = 0
        tsv = []
        tok_map = {}
        tok_count = 1
        prev_sent = 1
        cur_sent_subtok = 0

        # get token and sentence information
        toks = [y for x in example['sentences'] for y in x]
        for i in range(len(example['sentence_map'])):
            cur_subtok = example['subtoken_map'][i]
            cur_sent = example['sentence_map'][i] + 1
            if cur_sent != prev_sent:
                tok_count = 1
                prev_sent = cur_sent
                cur_sent_subtok = cur_subtok
            tok_map[i] = {'sent': cur_sent,
                          'tok': toks[i],
                          'sub_tok': example['subtoken_map'][i],
                          'tok_id': tok_count,
                          'sent_subtok': example['subtoken_map'][i] - cur_sent_subtok + 1
                          }
            tok_count += 1

        # generate mention spans and which span corefers to which one
        # coref {1:2, 2:3}
        # mentions {331:1, 332:1, 339:2}
        # mention_start {1:True}
        # mention_start_pos {1:331}
        predicted_clusters = example['predicted_clusters'] if 'predicted_clusters' in example else example['clusters']
        coref, mentions, mention_start, mention_start_pos = {}, {}, {}, {}
        mention_num = 1
        for cluster in predicted_clusters:
            cluster.sort()
            prev_mention_num = 0
            for m in cluster:
                start, end = m[0], m[-1]
                span = [x for x in range(int(start), int(end)+1)]
                for s in span:
                    if s not in mentions.keys():
                        mentions[s] = []
                    mentions[s].append(mention_num)

                if mention_num > prev_mention_num:
                    coref[prev_mention_num] = mention_num
                mention_start[mention_num] = True
                mention_start_pos[mention_num] = tok_map[start]
                prev_mention_num = mention_num
                mention_num += 1

        # generate tsv format
        prev_sent = '0'
        sent, tok_count = [], 0
        last_subtok = -1
        for i in tok_map.keys():
            if i == len(tok_map) - 1:
                a = 1
            cur_sent = tok_map[i]['sent']
            cur_tok = tok_map[i]['tok']
            cur_subtok = tok_map[i]['sub_tok']

            if cur_tok in ['[CLS]', '[SEP]']:
                continue
            elif cur_subtok == last_subtok:
                sent[-1][2] += cur_tok.replace('#', '')
                continue

            tok_count += 1
            cur_line = ['_', '_', cur_tok, '_', '_', '_', '_']
            if i in mentions.keys():
                for j in mentions[i]:
                    cur_line[-4] += f'abstract[{j}]|'
                    cur_line[-3] += f'new[{j}]|'
                    if mention_start[j]:
                        if j in coref.keys():
                            # find the coref next of j
                            next_mention_pos = mention_start_pos[coref[j]]

                            cur_line[-2] += f'coref|'
                            cur_line[-1] += f'{next_mention_pos["sent"]}-{next_mention_pos["sent_subtok"]}[{j}_{coref[j]}]|'
                            mention_start[j] = False
                for n in range(-4, 0):
                    cur_line[n] = cur_line[n].strip('_').strip('|') if cur_line[n] != '_' else '_'

            if cur_sent != prev_sent:
                tsv.append(sent)
                sent = []
                tok_count = 1
            cur_line[0] = f'{cur_sent}-{tok_count}'
            sent.append(cur_line)

            prev_sent = cur_sent
            last_subtok = cur_subtok

        # add the last sent to the list
        tsv.append(sent)

        text = '#FORMAT=WebAnno TSV 3.2\n#T_SP=webanno.custom.Referent|entity|infstat\n#T_RL=webanno.custom.Coref|type|BT_webanno.custom.Referent\n\n\n'
        for sent in tsv:
            sent_out = []

            tok_num = 1
            if not sent:
                continue
            for i, line in enumerate(sent):
                sent_out.append(line)

            sent_text = '#Text=' + ' '.join([i[2] for i in sent_out]) + '\n'
            text += sent_text + '\n'.join(['\t'.join(x) for x in sent_out]) + '\n\n'

        with io.open(out_dir+os.sep+docname+'.tsv', 'w') as f:
            f.write(text)
