import io, re, os
from glob import glob
import xmltodict

wikicoref_dir_path = "./WikiCoref/"


def read_xml(filename):
    with io.open(filename, 'r', encoding='utf8') as fd:
        file_dict = xmltodict.parse(fd.read())
    return file_dict


def convert_single_file(xml_file):
    token_file = xml_file.replace("/Markables/", "/Basedata/").replace("_coref_level_OntoNotesScheme.xml", "_words.xml")
    sentence_file = xml_file.replace("_coref_level_OntoNotesScheme.xml", "_sentence_level.xml")
    basename = "wikicoref/" + xml_file.split("/")[-1].replace("_coref_level_OntoNotesScheme.xml", "").replace(",", "_").replace(" ", "")
    token_lines = [x["#text"] for x in read_xml(token_file)["words"]["word"]]
    sentence_lines = [x["@span"].replace("word_", "").split("..") for x in read_xml(sentence_file)["markables"]["markable"]]
    xml_lines = [(x["@span"].replace("word_", "").split(".."), int(x["@coref_class"].replace("set_", ""))) for x in read_xml(xml_file)["markables"]["markable"]]
    xml_dict = {(int(x[0][0]), int(x[0][1])): x[1] for x in xml_lines}

    # create conll output
    single_file_conll_output = [[]]
    tok_id = 0
    doc_id = 0
    for sentence_line_id, sentence_line in enumerate(sentence_lines):
        single_sentence_conll_output = []
        start_tok, end_tok = int(sentence_line[0]), int(sentence_line[1])
        sent_len = end_tok - start_tok + 1

        if tok_id + sent_len > 1200 * (doc_id+1):
            doc_id += 1
            single_file_conll_output.append([])

        for sent_tok_id, tok in enumerate(range(start_tok, end_tok + 1)):
            coref_sets = []
            for xml_key in xml_dict.keys():
                if tok == xml_key[0] == xml_key[1]:
                    coref_sets.append("(%d)" % xml_dict[xml_key])
                elif tok == xml_key[0]:
                    coref_sets.append("(%d" % xml_dict[xml_key])
                elif tok == xml_key[1]:
                    coref_sets.append("%d)" % xml_dict[xml_key])
            coref_sets = sorted(coref_sets)

            line = [basename, f"{doc_id}", "%d" % sent_tok_id, token_lines[tok - 1]] + ["_"] * 6 + ["*"] * 6 + ["-"]
            if coref_sets:
                line[-1] = "|".join(coref_sets)
            single_sentence_conll_output.append(line)
            tok_id += 1

        single_file_conll_output[doc_id].append("\n".join(["\t".join(x) for x in single_sentence_conll_output]))

    # Finally, add meta data
    file_output = ""
    for doc_id, doc in enumerate(single_file_conll_output):
        file_output += f"#begin document (%s); part 00{doc_id}\n" % (basename) + "\n\n".join(doc) + "\n\n#end document\n"

    return file_output


if __name__ == '__main__':
    xml_files = sorted(glob(wikicoref_dir_path + "Annotation/**/Markables/*_coref_level_OntoNotesScheme.xml", recursive=True))
    wikicoref_output_file_path = "./wikicoref.v4_gold_conll"
    corpus_output = ""
    for xml_file in xml_files:
        corpus_output += convert_single_file(xml_file)

    with io.open(wikicoref_output_file_path, "w", encoding="utf8") as f:
        f.write(corpus_output)
