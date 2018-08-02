from ctcdecode import CTCBeamDecoder
import torch.nn.functional as F
import torch


class Decoder:

    def __init__(self, labels, lm_path=None, alpha=1, beta=1.5, cutoff_top_n=40, cutoff_prob=1.0, beam_width=200, num_processes=4, blank_index=0):
        self.vocab_list = labels + ['_'] # NOTE: blank symbol
        self._decoder = CTCBeamDecoder(self.vocab_list, lm_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_width, num_processes, blank_index)

    def convert_to_string(self, tokens, seq_len=None):
        if not seq_len:
            seq_len = tokens.size(0)
        out = []
        for i in range(seq_len):
            if len(out) == 0:
                if tokens[i] != 0:
                    out.append(tokens[i])
            else:
                if tokens[i] != 0 and tokens[i] != tokens[i - 1]:
                    out.append(tokens[i])
        return ''.join(self.vocab_list[i - 1] for i in out)
    
    def decode_beam(self, logits):
        decoded = []
        tlogits = logits.transpose(0, 1)
        beam_result, beam_scores, timesteps, out_seq_len = self._decoder.decode(tlogits)
        for i in range(tlogits.size(0)):
            output_str = self.convert_to_string(beam_result[i][0], out_seq_len[i][0])
            decoded.append(output_str)
        return decoded

    def decode_greedy(self, logits):
        decoded = []
        tlogits = logits.transpose(0, 1)
        _, tokens = torch.max(tlogits, 2)
        for i in range(tlogits.size(0)):
            output_str = self.convert_to_string(tokens[i])
            decoded.append(output_str)
        return decoded

