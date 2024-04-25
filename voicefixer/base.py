import librosa.display
from voicefixer.tools.pytorch_util import *
from voicefixer.tools.wav import *
from voicefixer.restorer.model import VoiceFixer as voicefixer_fe
import os
import time
# from torch.quantization import QuantStub, DeQuantStub
import torch.quantization as quantization
from torchsummary import summary
from prettytable import PrettyTable
import onnxruntime
import onnxscript
import librosa

import io


EPS = 1e-8


class VoiceFixer(nn.Module):
    def __init__(self):
        super(VoiceFixer, self).__init__()
        self._model = voicefixer_fe(channels=2, sample_rate=44100)

        # self._model.quant = QuantStub()
        # self._model.dequant = DeQuantStub()

        # self._model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        # quantization.prepare(self._model, inplace=True)

        # print(os.path.join(os.path.expanduser('~'), ".cache/voicefixer/analysis_module/checkpoints/epoch=15_trimed_bn.ckpt"))
        self.analysis_module_ckpt = os.path.join(
                    os.path.expanduser("~"),
                    ".cache/voicefixer/analysis_module/checkpoints/vf.ckpt",
        )
        if(not os.path.exists(self.analysis_module_ckpt)):
            raise RuntimeError("Error 0: The checkpoint for analysis module (vf.ckpt) is not found in ~/.cache/voicefixer/analysis_module/checkpoints. \
                                By default the checkpoint should be download automatically by this program. Something bad may happened.\
                                But don't worry! Alternatively you can download it directly from Zenodo: https://zenodo.org/record/5600188/files/vf.ckpt?download=1.")
        saved_state_dict = torch.load(self.analysis_module_ckpt)
        model_state_dict = self._model.state_dict()

        new_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_state_dict}

        model_state_dict.update(new_state_dict)
        self._model.load_state_dict(model_state_dict, strict=False)
        self._model.eval()

        # self_model = quantization.convert(self._model, inplace=False)
        # print("model is quantized now static")
        # print(self._model)

        # self._model = quantization.quantize_dynamic(self._model, {torch.nn.Linear}, dtype=torch.qint8)
        # print(self._model)
        # print("model is quantizes now dynamic")

    def _load_wav_energy(self, path, sample_rate, threshold=0.95):
        wav_10k, _ = librosa.load(path, sr=sample_rate)
        stft = np.log10(np.abs(librosa.stft(wav_10k)) + 1.0)
        fbins = stft.shape[0]
        e_stft = np.sum(stft, axis=1)
        for i in range(e_stft.shape[0]):
            e_stft[-i - 1] = np.sum(e_stft[: -i - 1])
        total = e_stft[-1]
        for i in range(e_stft.shape[0]):
            if e_stft[i] < total * threshold:
                continue
            else:
                break
        return wav_10k, int((sample_rate // 2) * (i / fbins))

    def _load_wav(self, path, sample_rate, threshold=0.95):
        wav_10k, _ = librosa.load(path, sr=sample_rate)
        return wav_10k

    def _amp_to_original_f(self, mel_sp_est, mel_sp_target, cutoff=0.2):
        freq_dim = mel_sp_target.size()[-1]
        mel_sp_est_low, mel_sp_target_low = (
            mel_sp_est[..., 5 : int(freq_dim * cutoff)],
            mel_sp_target[..., 5 : int(freq_dim * cutoff)],
        )
        energy_est, energy_target = torch.mean(mel_sp_est_low, dim=(2, 3)), torch.mean(
            mel_sp_target_low, dim=(2, 3)
        )
        amp_ratio = energy_target / energy_est
        return mel_sp_est * amp_ratio[..., None, None], mel_sp_target

    def _trim_center(self, est, ref):
        diff = np.abs(est.shape[-1] - ref.shape[-1])
        if est.shape[-1] == ref.shape[-1]:
            return est, ref
        elif est.shape[-1] > ref.shape[-1]:
            min_len = min(est.shape[-1], ref.shape[-1])
            est, ref = est[..., int(diff // 2) : -int(diff // 2)], ref
            est, ref = est[..., :min_len], ref[..., :min_len]
            return est, ref
        else:
            min_len = min(est.shape[-1], ref.shape[-1])
            est, ref = est, ref[..., int(diff // 2) : -int(diff // 2)]
            est, ref = est[..., :min_len], ref[..., :min_len]
            return est, ref

    def _pre(self, model, input, cuda):
        input = input[None, None, ...]
        input = torch.tensor(input)
        input = try_tensor_cuda(input, cuda=cuda)
        sp, _, _ = model.f_helper.wav_to_spectrogram_phase(input)
        mel_orig = model.mel(sp.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        # return models.to_log(sp), models.to_log(mel_orig)
        return sp, mel_orig

    def remove_higher_frequency(self, wav, ratio=0.95):
        stft = librosa.stft(wav)
        real, img = np.real(stft), np.imag(stft)
        mag = (real**2 + img**2) ** 0.5
        cos, sin = real / (mag + EPS), img / (mag + EPS)
        spec = np.abs(stft)  # [1025,T]
        feature = spec.copy()
        feature = np.log10(feature + EPS)
        feature[feature < 0] = 0
        energy_level = np.sum(feature, axis=1)
        threshold = np.sum(energy_level) * ratio
        curent_level, i = energy_level[0], 0
        while i < energy_level.shape[0] and curent_level < threshold:
            curent_level += energy_level[i + 1, ...]
            i += 1
        spec[i:, ...] = np.zeros_like(spec[i:, ...])
        stft = spec * cos + 1j * spec * sin
        return librosa.istft(stft)

    # def convert_to_onnx(self, input_shape, output_file):
    #     # Create a dummy input tensor with the specified shape
    #     dummy_input = torch.randn(*input_shape)
    #
    #     # Export the model to ONNX format
    #     torch.onnx.export(self._model, dummy_input, output_file, verbose=True)


    @torch.no_grad()
    def restore_inmem(self, wav_10k, cuda=False, mode=0, your_vocoder_func=None, ):
        check_cuda_availability(cuda=cuda)
        self._model = try_tensor_cuda(self._model, cuda=cuda)
        if mode == 0:
            self._model.eval()
        elif mode == 1:
            self._model.eval()
        elif mode == 2:
            self._model.train()  # More effective on seriously demaged speech
        res = []
        seg_length = int(44100 * 1)
        # print("seg_length", seg_length)
        break_point = seg_length

        # sess = onnxruntime.InferenceSession("myvocoder.onnx", providers=['CPUExecutionProvider'])
        # input_name = sess.get_inputs()[0].name
        # print("input name", input_name)
        # input_shape = sess.get_inputs()[0].shape
        # print("input shape", input_shape)
        # input_type = sess.get_inputs()[0].type
        # print("input type", input_type)


        while break_point < wav_10k.shape[0] + seg_length:
            segment = wav_10k[break_point - seg_length : break_point]
            # if len(segment) < seg_length:
            #     # Pad with zeros if the segment is shorter than seg_length
            #     pad_length = seg_length - len(segment)
            #     segment = np.concatenate((np.zeros(pad_length), segment))

            # print("segment", segment.shape)
            # If mode is 1, remove higher frequency components from the segment
            if mode == 1:
                segment = self.remove_higher_frequency(segment)
            s = time.time()
            sp, mel_noisy = self._pre(self._model, segment, cuda)
            print("first", time.time() - s)

            # mel_orig = mel_noisy

            # torch.onnx.export(self._model, sp, mel_orig,onnx_file_path, verbose=True)

            # onnx_file_path = "my.onnx"
            # if not os.path.exists(onnx_file_path):
            #     onnx_program = torch.onnx.export(self._model, (sp, mel_orig), "my.onnx")
            #     print("converted")
            # ort_inputs = {
            #     "mel_orig": mel_noisy.detach().numpy()
            # }
            # out_model = sess.run(None, ort_inputs)


            # print(type(out_model))

            s = time.time()
            out_model = self._model(sp, mel_noisy)
            print("second", time.time() - s)

            # print(type(out_model))
            # out_model = torch.tensor(out_model[0])
            denoised_mel = from_log(out_model['mel'])

            if not os.path.exists("mel"):
                os.makedirs("mel")

            # Save the denoised mel spectrogram in memory
            f = io.BytesIO()
            torch.save(denoised_mel, f, _use_new_zipfile_serialization=True)

            # Write the bytes to file
            saved_denoised_mel_path = os.path.join("mel", f"denoised_mel_{break_point}.pt")
            with open(saved_denoised_mel_path, 'wb') as file:
                file.write(f.getvalue())

            # Load the denoised mel spectrogram
            with open(saved_denoised_mel_path, 'rb') as file:
                loaded_denoised_mel = torch.load(file)


            if your_vocoder_func is None:
                start_time = time.time()

                out = self._model.vocoder(loaded_denoised_mel, cuda=cuda)

                # ort_inputs = {
                #         "mel": denoised_mel.detach().numpy()
                #     }
                # out = sess.run(None, ort_inputs)
                # print(out)
                # out = torch.tensor(out)
                # onnx_file_path = "myvocoder.onnx"
                # if not os.path.exists(onnx_file_path):
                #     onnx_program = torch.onnx.export(self._model.vocoder, (denoised_mel), "myvocoder.onnx")
                #     print("converted")
                end_time = time.time()
                print("Third time taken", end_time - start_time)
            else:
                out = your_vocoder_func(denoised_mel)

            # unify energy
            if torch.max(torch.abs(out)) > 1.0:
                out = out / torch.max(torch.abs(out))
                print("Warning: Exceed energy limit,", input)
            # frame alignment
            out, _ = self._trim_center(out, segment)
            res.append(out)
            break_point += seg_length

        out = torch.cat(res, -1)
        return tensor2numpy(out.squeeze(0))


    def restore(self, input, output, cuda=False, mode=0, your_vocoder_func=None):
        wav_10k = self._load_wav(input, sample_rate=44100)
        out_np_wav = self.restore_inmem(
            wav_10k, cuda=cuda, mode=mode, your_vocoder_func=your_vocoder_func
        )
        save_wave(out_np_wav, fname=output, sample_rate=44100)

    def new_restore(self, input, cuda=False, mode=0, your_vocoder_func=None):
        wav_10k = input
        # print("hi")
        out_np_wav = self.restore_inmem(
            wav_10k, cuda=cuda, mode=mode, your_vocoder_func=your_vocoder_func
        )
        return out_np_wav

