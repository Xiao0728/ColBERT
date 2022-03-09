import torch

from transformers import BertTokenizerFast
from colbert.modeling.tokenization.utils import _split_into_batches


class QueryTokenizer():
    def __init__(self, query_maxlen):
        self.tok = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.query_maxlen = query_maxlen

        self.Q_marker_token, self.Q_marker_token_id = '[Q]', self.tok.convert_tokens_to_ids('[unused0]')
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id
        self.mask_token, self.mask_token_id = self.tok.mask_token, self.tok.mask_token_id

        assert self.Q_marker_token_id == 1 and self.mask_token_id == 103

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.Q_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix + [self.mask_token] * (self.query_maxlen - (len(lst)+3)) for lst in tokens]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        ids = self.tok(batch_text, add_special_tokens=False)['input_ids']

        if not add_special_tokens:
            return ids

        prefix, suffix = [self.cls_token_id, self.Q_marker_token_id], [self.sep_token_id]
        ids = [prefix + lst + suffix + [self.mask_token_id] * (self.query_maxlen - (len(lst)+3)) for lst in ids]

        return ids

    def tensorize(self, batch_text, bsize=None):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        # add placehold for the [Q] marker
        
        batch_text = ['. ' + x for x in batch_text]

        obj = self.tok(batch_text, padding='max_length', truncation=True,
                       return_tensors='pt', max_length=self.query_maxlen)

        ids, mask = obj['input_ids'], obj['attention_mask']

        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[:, 1] = self.Q_marker_token_id
        ids[ids == 0] = self.mask_token_id

        if bsize:
            batches = _split_into_batches(ids, mask, bsize)
            return batches

        return ids, mask

class PRFQueryTokenizer(QueryTokenizer):

    def __init__(self,query_maxlen, num_prf=3):
        super().__init__(query_maxlen=query_maxlen)
        self.num_prf = num_prf

    # def __init__(self, query_maxlen, num_prf=3):
    #     self.tok = BertTokenizerFast.from_pretrained('bert-base-uncased')
    #     self.query_maxlen = query_maxlen
    #     self.Q_marker_token, self.Q_marker_token_id = '[Q]', self.tok.convert_tokens_to_ids('[unused0]')
    #     self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
    #     self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id
    #     self.mask_token, self.mask_token_id = self.tok.mask_token, self.tok.mask_token_id
    #     self.num_prf = num_prf
    #     assert self.Q_marker_token_id == 1 and self.mask_token_id == 103
    

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))
        prefix, suffix = [self.cls_token, self.Q_marker_token], [self.sep_token]
        
        tokens = []
        prftokenlst = []
        for prfquery in batch_text:
            text = prfquery.strip().split(" \t ")
            qtokenlst = self.tok.tokenize(text[0],add_special_tokens=False)
            for i in range(1,self.num_prf):
                prftoken = suffix + self.tok.tokenize(text[i],add_special_tokens=False)
                prftokenlst.extend(prftoken)
            tokensOneline = prefix + qtokenlst  + prftokenlst+ suffix + [self.tok.mask_token] * (self.query_maxlen - (len(qtokenlst)+len(prftokenlst)+3))
            tokens.append(tokensOneline)
        return tokens

    def encode(self, batch_text, add_speical_tokens = False):
        assert type(batch_text) in [list, tuple], (type(batch_text)) 
        prefix, suffix = [self.cls_token_id, self.Q_marker_token_id], [self.sep_token_id] 
        ids = []
        prfidslst = []
        for prfquery in batch_text:
            text = prfquery.strip().split(" \t ") 
            query_ids = self.tok(text[0], add_special_tokens=False)['input_ids'] 
            for i in range(1,self.num_prf):
                prfids = suffix + self.tok(text[i], add_speical_tokens=False)['input_ids']
                prfidslst.extend(prfids)

            idsOneline = prefix + query_ids + prfidslst + suffix + [self.mask_token_id] *(self.query_maxlen - (len(query_ids)+len(prfidslst)+3)) 

            ids.append(idsOneline)
        return ids

    # def tensorize(self, batch_text, bsize=None):
    #     assert type(batch_text) in [list, tuple], (type(batch_text))

    #     # add placehold for the [Q] marker
    #     batch_text = ['. ' + x for x in batch_text]

    #     obj = self.tok(batch_text, padding='max_length', truncation=True,
    #                    return_tensors='pt', max_length=self.query_maxlen)

    #     ids, mask = obj['input_ids'], obj['attention_mask']

    #     # postprocess for the [Q] marker and the [MASK] augmentation
    #     ids[:, 1] = self.Q_marker_token_id
    #     ids[ids == 0] = self.mask_token_id

    #     if bsize:
    #         batches = _split_into_batches(ids, mask, bsize)
    #         return batches

    #     return ids, mask
