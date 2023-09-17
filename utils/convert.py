from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import re
import os
import json
import collections
import argparse
import conll
from transformers import BertTokenizer


DATA_FLAG = '_sg'
entity_class = {'person': 1, 'place': 2, 'organization': 3, 'object': 4, 'event': 5, 'time': 6, 'substance': 7, 'animal': 8, 'plant': 9, 'abstract': 10}
infstat_class = {'new': 1, 'giv_act': 2, 'giv_inact': 3, 'acc_com': 4, 'acc_inf': 5, 'acc_aggr': 6, 'undefined': 7}

class DocumentState(object):
    def __init__(self, key, do_discourse):
        self.doc_key = key
        self.do_discourse = do_discourse
        self.sentence_end = []
        self.token_end = []
        self.tokens = []
        self.subtokens = []
        self.info = []
        self.segments = []
        self.subtoken_map = []
        self.segment_subtoken_map = []
        self.sentence_map = []
        self.pronouns = []
        self.entities = collections.defaultdict(list)
        self.discourse_stacks = collections.defaultdict(list)
        self.clusters = collections.defaultdict(list)
        self.coref_stacks = collections.defaultdict(list)
        self.infstats = collections.defaultdict(list)
        self.infstat_stacks = collections.defaultdict(list)
        self.speakers = []
        self.segment_info = []

    def finalize(self):
        # finalized: segments, segment_subtoken_map
        # populate speakers from info
        subtoken_idx = 0
        for segment in self.segment_info:
            speakers = []
            for i, tok_info in enumerate(segment):
                if tok_info is None and (i == 0 or i == len(segment) - 1):
                    speakers.append('[SPL]')
                elif tok_info is None:
                    speakers.append(speakers[-1])
                else:
                    speakers.append(tok_info[9])
                    if tok_info[4] == 'PRP':
                        self.pronouns.append(subtoken_idx)
                subtoken_idx += 1
            self.speakers += [speakers]
        # populate sentence map

        # populate clusters
        first_coref_subtoken_index = -1
        first_entity_subtoken_index = -1
        for seg_idx, segment in enumerate(self.segment_info):
            speakers = []
            for i, tok_info in enumerate(segment):
                first_coref_subtoken_index += 1
                first_entity_subtoken_index += 1
                coref = tok_info[-2] if tok_info is not None else '-'
                if coref != "-":
                    last_coref_subtoken_index = first_coref_subtoken_index + tok_info[-1] - 1
                    for part in coref.split("|"):
                        if part[0] == "(":
                            if part[-1] == ")":
                                cluster_id = int(part[1:-1])
                                self.clusters[cluster_id].append((first_coref_subtoken_index, last_coref_subtoken_index))
                            else:
                                cluster_id = int(part[1:])
                                self.coref_stacks[cluster_id].append(first_coref_subtoken_index)
                        else:
                            if not part[:-1]:
                                a = 1
                            assert part[:-1], "\t".join([str(info) for info in tok_info])
                            cluster_id = int(part[:-1])
                            assert self.coref_stacks[cluster_id], "\t".join([str(info) for info in tok_info])
                            start = self.coref_stacks[cluster_id].pop()
                            self.clusters[cluster_id].append((start, last_coref_subtoken_index))

                    # entity
                    if self.do_discourse:
                        entity = tok_info[-8] if tok_info else '*'
                        # if entity == '*':
                        #     continue
                        assert entity != "*", "\t".join([str(info) for info in tok_info])
                        if len(coref.split("|")) != len(entity.split("|")):
                            continue
                        last_entity_subtoken_index = first_entity_subtoken_index + tok_info[-1] - 1
                        for part in entity.split("|"):
                            e_id = part.strip("(").strip(")").split("-")[0]
                            if part[0] == "(":
                                e_type = part.strip("(").strip(")").split("-")[1]
                                e_type_class = entity_class[e_type]
                                infs = part.strip("(").strip(")").split("-")[2]
                                infs_class = infstat_class[infs]
                                if part[-1] == ")":
                                    self.entities[e_id].append((first_entity_subtoken_index, last_entity_subtoken_index, e_type_class))
                                    self.infstats[e_id].append((first_entity_subtoken_index, last_entity_subtoken_index, infs_class))
                                else:
                                    self.discourse_stacks[e_id].append((first_entity_subtoken_index, e_type_class, infs_class))
                            else:
                                if e_id not in self.discourse_stacks:
                                    continue
                                e_start, e_type_class, infs_class = self.discourse_stacks[e_id].pop()
                                self.entities[e_id].append((e_start, last_entity_subtoken_index, e_type_class))
                                self.infstats[e_id].append((e_start, last_entity_subtoken_index, infs_class))

                        a = 1

        # merge clusters
        merged_clusters = []
        for c1 in self.clusters.values():
            existing = None
            for m in c1:
                for c2 in merged_clusters:
                    if m in c2:
                        existing = c2
                        break
                if existing is not None:
                    break
            if existing is not None:
                #         print(c1)
                #         print(self.doc_key)
                print("Merging clusters (shouldn't happen very often.)")
                existing.update(c1)
            else:
                merged_clusters.append(set(c1))

        # remove singletons
        new_clusters = []
        for cluster in merged_clusters:
            if len(cluster) < 2:
                continue
            new_clusters.append(list(cluster))

        # merged_clusters = [list(c) for c in merged_clusters]
        #     print(merged_clusters)
        all_mentions = flatten(new_clusters)
        sentence_map = get_sentence_map(self.segments, self.sentence_end)
        subtoken_map = flatten(self.segment_subtoken_map)
        #     print(len(all_mentions))
        #     print(len(set(all_mentions)))
        assert len(all_mentions) == len(set(all_mentions))

        num_words = len(flatten(self.segments))
        assert num_words == len(flatten(self.speakers))
        assert num_words == len(subtoken_map), (num_words, len(subtoken_map))
        assert num_words == len(sentence_map), (num_words, len(sentence_map))
        return {
            "doc_key": self.doc_key,
            "sentences": self.segments,
            "speakers": self.speakers,
            "constituents": [],
            "entity": [v[0] for v in self.entities.values()],
            "infstat": [v[0] for v in self.infstats.values()],
            "clusters": new_clusters,
            'sentence_map': sentence_map,
            "subtoken_map": subtoken_map,
            'pronouns': self.pronouns
        }

def flatten(l):
    return [item for sublist in l for item in sublist]

def normalize_word(word, language):
    if language == "arabic":
        word = word[:word.find("#")]
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word

# first try to satisfy constraints1, and if not possible, constraints2.
def split_into_segments(document_state, max_segment_len, constraints1, constraints2):
    current = 0
    previous_token = 0
    while current < len(document_state.subtokens):
        end = min(current + max_segment_len - 1 - 2, len(document_state.subtokens) - 1)
        while end >= current and not constraints1[end]:
            end -= 1
        if end < current:
            end = min(current + max_segment_len - 1 - 2, len(document_state.subtokens) - 1)
            while end >= current and not constraints2[end]:
                end -= 1
            if end < current:
                raise Exception("Can't find valid segment")
        document_state.segments.append(['[CLS]'] + document_state.subtokens[current:end + 1] + ['[SEP]'])
        subtoken_map = document_state.subtoken_map[current : end + 1]
        document_state.segment_subtoken_map.append([previous_token] + subtoken_map + [subtoken_map[-1]])
        info = document_state.info[current : end + 1]
        document_state.segment_info.append([None] + info + [None])
        current = end + 1
        previous_token = subtoken_map[-1]

def get_sentence_map(segments, sentence_end):
    current = 0
    sent_map = []
    sent_end_idx = 0
    assert len(sentence_end) == sum([len(s) -2 for s in segments])
    for segment in segments:
        sent_map.append(current)
        for i in range(len(segment) - 2):
            sent_map.append(current)
            current += int(sentence_end[sent_end_idx])
            sent_end_idx += 1
        sent_map.append(current)
    return sent_map

def get_document(document_lines, tokenizer, language, segment_len, do_discourse):
    document_state = DocumentState(document_lines[0], do_discourse)
    word_idx = -1
    for line in document_lines[1]:
        row = line.split()
        sentence_end = len(row) == 0
        if not sentence_end:
            #       if len(row) >= 12:
            #         print(row)
            assert len(row) >= 12
            word_idx += 1
            word = normalize_word(row[3], language)
            subtokens = tokenizer.tokenize(word)
            document_state.tokens.append(word)
            document_state.token_end += ([False] * (len(subtokens) - 1)) + [True]
            for sidx, subtoken in enumerate(subtokens):
                document_state.subtokens.append(subtoken)
                info = None if sidx != 0 else (row + [len(subtokens)])
                document_state.info.append(info)
                document_state.sentence_end.append(False)
                document_state.subtoken_map.append(word_idx)
        else:
            document_state.sentence_end[-1] = True
    # split_into_segments(document_state, segment_len, document_state.token_end)
    # split_into_segments(document_state, segment_len, document_state.sentence_end)
    constraints1 = document_state.sentence_end if language != 'arabic' else document_state.token_end
    split_into_segments(document_state, segment_len, constraints1, document_state.token_end)
    stats["max_sent_len_{}".format(language)] = max(max([len(s) for s in document_state.segments]), stats["max_sent_len_{}".format(language)])
    document = document_state.finalize()
    return document

def skip(doc_key):
    # if doc_key in ['nw/xinhua/00/chtb_0078_0', 'wb/eng/00/eng_0004_1']: #, 'nw/xinhua/01/chtb_0194_0', 'nw/xinhua/01/chtb_0157_0']:
    # return True
    return False

def minimize_partition(name, language, extension, labels, stats, tokenizer, seg_len, input_dir, output_dir, do_discourse):
    # input_path = "{}/{}.{}.{}".format(input_dir, name, language, extension)
    input_path = os.path.join(input_dir, name)
    output_path = os.path.join(output_dir, '.'.join(name.split('.')[:-1])+f'.{seg_len}.jsonlines')
    # output_path = "{}/{}.{}.{}.jsonlines".format(output_dir, name, language, seg_len)
    count = 0
    print("Minimizing {}".format(input_path))
    documents = []
    with open(input_path, "r", encoding='utf-8') as input_file:
        for line in input_file.readlines():
            begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
            if begin_document_match:
                doc_key = conll.get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
                documents.append((doc_key, []))
            elif line.startswith("#end document"):
                continue
            else:
                documents[-1][1].append(line)
    with open(output_path, "w", encoding='utf-8') as output_file:
        for document_lines in documents:
            if skip(document_lines[0]):
                continue
            if 'reddit/conspiracy' in document_lines[0]:
                a = 1
            document = get_document(document_lines, tokenizer, language, seg_len, do_discourse)
            output_file.write(json.dumps(document))
            output_file.write("\n")
            count += 1
    print("Wrote {} documents to {}".format(count, output_path))

def minimize_language(language, labels, stats, model, seg_len, input_dir, output_dir, do_lower_case, do_discourse):
    # do_lower_case = True if 'chinese' in vocab_file else False
    tokenizer = BertTokenizer.from_pretrained(model)

    for filename in os.listdir(input_dir):
        if not filename.endswith('v4_gold_conll'):
            continue
        minimize_partition(filename, language, "v4_gold_conll", labels, stats, tokenizer, seg_len, input_dir, output_dir, do_discourse)
    # minimize_partition(f"train{DATA_FLAG}", language, "v4_gold_conll", labels, stats, tokenizer, seg_len, input_dir, output_dir)
    # minimize_partition(f"test{DATA_FLAG}", language, "v4_gold_conll", labels, stats, tokenizer, seg_len, input_dir, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--model", default="bert-base-cased")
    parser.add_argument("-i", "--input", default="./data/ontogum_sg")
    parser.add_argument("-o", "--output", default="./data/ontogum_sg")
    parser.add_argument("--lower", action="store_true")
    parser.add_argument("--discourse", action="store_true")

    args = parser.parse_args()
    model = args.model
    input_dir = args.input
    output_dir = args.output
    do_lower_case = args.lower
    do_discourse = args.discourse

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(do_lower_case)
    labels = collections.defaultdict(set)
    stats = collections.defaultdict(int)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for seg_len in [128, 256, 384, 512]:
        minimize_language("english", labels, stats, model, seg_len, input_dir, output_dir, do_lower_case, do_discourse)
        # minimize_language("chinese", labels, stats, vocab_file, seg_len)
        # minimize_language("es", labels, stats, vocab_file, seg_len)
        # minimize_language("arabic", labels, stats, vocab_file, seg_len)
    for k, v in labels.items():
        print("{} = [{}]".format(k, ", ".join("\"{}\"".format(label) for label in v)))
    for k, v in stats.items():
        print("{} = {}".format(k, v))
