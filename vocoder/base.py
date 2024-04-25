from voicefixer.vocoder.model.generator import Generator
from voicefixer.tools.wav import read_wave, save_wave
from voicefixer.tools.pytorch_util import *
from voicefixer.vocoder.model.util import *
from voicefixer.vocoder.config import Config
import os
import numpy as np
import time
from torch.nn.parallel import DataParallel
import torch.quantization
import multiprocessing
import openvino as ov
import time
from nncf import compress_weights, CompressWeightsMode
import torch.nn.utils.prune as prune
import io
import onnxruntime
import onnxscript
from torchsummary import summary
import onnx

class Vocoder(nn.Module):
    def __init__(self, sample_rate):
        super(Vocoder, self).__init__()
        Config.refresh(sample_rate)
        self.rate = sample_rate
        if(not os.path.exists(Config.ckpt)):
            raise RuntimeError("Error 1: The checkpoint for synthesis module / vocoder (model.ckpt-1490000_trimed) is not found in ~/.cache/voicefixer/synthesis_module/44100. \
                                By default the checkpoint should be download automatically by this program. Something bad may happened. Apologies for the inconvenience.\
                                But don't worry! Alternatively you can download it directly from Zenodo: https://zenodo.org/record/5600188/files/model.ckpt-1490000_trimed.pt?download=1")
        self._load_pretrain(Config.ckpt)
        self.weight_torch = Config.get_mel_weight_torch(percent=1.0)[
            None, None, None, ...
        ]
        self.count = 0
        core = ov.Core()
        model = core.read_model('model.xml', weights='model.bin')
        num_cores = os.cpu_count()
        # print(num_cores)
        self.compiled_model = core.compile_model(model=model, device_name="CPU", config={"INFERENCE_NUM_THREADS": num_cores})



    def _load_pretrain(self, pth):
        self.model = Generator(Config.cin_channels)
        checkpoint = load_checkpoint(pth, torch.device("cpu"))
        load_try(checkpoint["generator"], self.model)
        self.model.eval()
        self.model.remove_weight_norm()
        self.model.remove_weight_norm()
        example_input = torch.randn(1, 128, 101)
        traced_script_module = torch.jit.trace(self.model, example_input)
        traced_script_module.save("torchscript_model.pt")
        print("torchscript_model saved")
        for p in self.model.parameters():
            p.requires_grad = False


    # def vocoder_mel_npy(self, mel, save_dir, sample_rate, gain):
    #     mel = mel / Config.get_mel_weight(percent=gain)[...,None]
    #     mel = normalize(amp_to_db(np.abs(mel)) - 20)
    #     mel = pre(np.transpose(mel, (1, 0)))
    #     with torch.no_grad():
    #         wav_re = self.model(mel) # torch.Size([1, 1, 104076])
    #         save_wave(tensor2numpy(wav_re)*2**15,save_dir,sample_rate=sample_rate)


    def _process_chunk(self, mel_chunk):
        """
        Process a single chunk of mel spectrogram using the model.
        """
        with torch.no_grad():
            wav_chunk = self.model(mel_chunk)
        return wav_chunk

    def convert_to_onnx(self, input_shape, output_file):
        # Create a dummy input tensor with the specified shape
        dummy_input = torch.randn(*input_shape)

        # Export the model to ONNX format
        torch.onnx.export(self._model, dummy_input, output_file, verbose=True)

    def forward(self, mel, cuda=False):
        """
        :param non normalized mel spectrogram: [batchsize, 1, t-steps, n_mel]
        :return: [batchsize, 1, samples]
        """
        # print(mel.size())

        assert mel.size()[-1] == 128
        check_cuda_availability(cuda=cuda)
        self.model = try_tensor_cuda(self.model, cuda=cuda)
        mel = try_tensor_cuda(mel, cuda=cuda)
        self.weight_torch = self.weight_torch.type_as(mel)
        mel = mel / self.weight_torch
        mel = tr_normalize(tr_amp_to_db(torch.abs(mel)) - 20.0)
        mel = tr_pre(mel[:, 0, ...])

        # onnx_file_path = "new_vocoder.onnx"

        # Print all layer names
        # for name, param in self.model.named_parameters():
        #     # print(name)

        # Load the ONNX model
        # model = onnx.load("new_vocoder.onnx")
        # Print input names
        # print("Input Names:")
        # for input in model.graph.input:
        #     print("\t", input.name)
        # # Print output names
        # print("Output Names:")
        # for output in model.graph.output:
        #     print("\t", output.name)

        # if not os.path.exists(onnx_file_path):
        #     summary(self.model, mel.shape[1:])
        #
        #     # for name, param in self.model.named_parameters():
        #     #     if '.' in name:
        #     #         first_layer_name = name.split('.')[0]
        #     #         # print("Name of the first layer:", first_layer_name)
        #     #         if 'condnet' in name:
        #     #             # print(f"Layer Name: {name}")
        #     #             # print(f"Weights:\n{param.data}")
        #     #             # print(f"Biases:\n{param.data}")
        #     #         break  # Stop after printing the first layer name
        #
        #     onnx_program = torch.onnx.export(self.model, (mel), "new_vocoder.onnx")
        #     print("converted")


        # if not os.path.exists("new_mel"):
        #     os.makedirs("new_mel")
        # f = io.BytesIO()
        # torch.save(mel, f, _use_new_zipfile_serialization=True)
        # # Write the bytes to file
        # saved_denoised_mel_path = os.path.join("new_mel", f"denoised_mel_{self.count}.pt")
        # with open(saved_denoised_mel_path, 'wb') as file:
        #     file.write(f.getvalue())
        # self.count = self.count+1
        # print("saved new mel: ", self.count)


        # self.model.eval()
        # ov_model = ov.convert_model(self.model, example_input=mel)
        # ov.save_model(ov_model, 'model.xml', compress_to_fp16=True)
        # print("model saved ")

        # core = ov.Core()
        # compiled_model = core.compile_model(ov_model)
        with torch.no_grad():
            # wav_re =  torch.from_numpy(self.compiled_model(mel)[0])
            wav_re = self.model(mel)

        # infer_request.infer()
        # input_tensor = infer_request.get_input_tensor()
        # output_tensor = infer_request.get_output_tensor()
        # infer_request = compiled_model.create_infer_request()
        # input_tensor = infer_request.get_tensor("mel")
        # infer_request.set_input_tensor(input_tensor)
        # infer_request.infer()
        # output_tensor = infer_request.get_output_tensor()
        # new_wav_re = torch.from_numpy(output_tensor.data)

        return wav_re



    def oracle(self, fpath, out_path, cuda=False):
        check_cuda_availability(cuda=cuda)
        self.model = try_tensor_cuda(self.model, cuda=cuda)
        wav = read_wave(fpath, sample_rate=self.rate)[..., 0]
        wav = wav / np.max(np.abs(wav))
        stft = np.abs(
            librosa.stft(
                wav,
                hop_length=Config.hop_length,
                win_length=Config.win_size,
                n_fft=Config.n_fft,
            )
        )
        mel = linear_to_mel(stft)
        mel = normalize(amp_to_db(np.abs(mel)) - 20)
        mel = pre(np.transpose(mel, (1, 0)))
        mel = try_tensor_cuda(mel, cuda=cuda)
        with torch.no_grad():
            wav_re = self.model(mel)
            save_wave(tensor2numpy(wav_re * 2**15), out_path, sample_rate=self.rate)


if __name__ == "__main__":
    model = Vocoder(sample_rate=44100)
    print(model.device)
    # model.load_pretrain(Config.ckpt)
    # model.oracle(path="/Users/liuhaohe/Desktop/test.wav",
    #         sample_rate=44100,
    #         save_dir="/Users/liuhaohe/Desktop/test_vocoder.wav")
