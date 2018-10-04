import argparse
import json
import os.path
import re
import nltk
import uuid
import csv

def add_arguments(parser):
    parser.add_argument("--format", help="format to generate", required=True)
    parser.add_argument("--input_file", help="path to input file", required=True)
    parser.add_argument("--output_file", help="path to output file", required=True)

def normalize_text(text, lower_case=True, remove_punc=False):
    def process_token(tokens):
        special = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        pattern = "([{}])".format("".join(special))
        processed_tokens = []
        for token in tokens:
            token = token.replace("''", '"').replace("``", '"')
            processed_tokens.extend(re.split(pattern, token))
        
        return processed_tokens
    
    def remove_punctuation(tokens):
        exclude = set(string.punctuation)
        return [token for token in tokens if token not in exclude]
    
    def fix_white_space(tokens):
        return [token for token in tokens if token and not token.isspace()]
    
    sents = nltk.sent_tokenize(text)
    norm_sents = []
    for sent in sents:
        words = nltk.word_tokenize(sent)
        words = process_token(words)
        if remove_punc:
            words = remove_punctuation(words)
        
        words = fix_white_space(words)
        norm_sents.append(' '.join(words))
    
    norm_text = ' '.join(norm_sents)
    if lower_case:
        norm_text = norm_text.lower()
    
    return norm_text

def preprocess(file_name):
    if not os.path.exists(file_name):
        raise FileNotFoundError("file not found")
    
    context_lookup = {}
    with open(file_name, "r", encoding="utf8") as file:
        file_reader = csv.reader(file, delimiter=',', quotechar='"')
        
        subitem_separator = "__EOS__"
        subitem_connector = "|"
        for items in file_reader:
            if len(items) < 3:
                continue
            
            context = [normalize_text(sentence) for sentence in items[0].split(subitem_separator)]
            context_key = subitem_connector.join(context)
            if context_key not in context_lookup:
                context_lookup[context_key] = {
                    "id": str(uuid.uuid4()),
                    "context": context,
                    "response": []
                }
            
            response = {
                "text": normalize_text(items[1]),
                "label": items[2]
            }
            
            context_lookup[context_key]["response"].append(response)
    
    processed_data_list = list(context_lookup.values())
    return processed_data_list

def output_to_json(data_list, file_name):
    with open(file_name, "w") as file:
        data_json = json.dumps(data_list, indent=4)
        file.write(data_json)

def output_to_plain(data_list, file_name):
    with open(file_name, "wb") as file:
        for data in data_list:
            for response in data["response"]:
                data_plain = "{0}\t{1}\t{2}\t{3}\r\n".format(data["id"], data["context"], response["text"], response["label"])
                file.write(data_plain.encode("utf-8"))

def main(args):
    processed_data = preprocess(args.input_file)
    if (args.format == 'json'):
        output_to_json(processed_data, args.output_file)
    elif (args.format == 'plain'):
        output_to_plain(processed_data, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)