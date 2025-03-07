# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import asyncio  # 新增异步支持
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{ROOT_DIR}/../../..")
sys.path.append(f"{ROOT_DIR}/../../../third_party/Matcha-TTS")
import logging
import argparse
import torchaudio
import cosyvoice_pb2
import cosyvoice_pb2_grpc
import grpc
from grpc import aio  # 使用异步grpc模块
from cosyvoice.utils.file_utils import load_wav

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import torch
import numpy as np


def convert_audio_bytes_to_tensor(raw_audio: bytes) -> torch.Tensor:
    """同步音频转换方法"""
    return torch.from_numpy(
        np.array(np.frombuffer(raw_audio, dtype=np.float32))
    ).unsqueeze(0)


def convert_audio_tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """同步Tensor转字节方法"""
    return tensor.numpy().astype(np.float32).tobytes()


async def main(args):
    # 使用异步通道和上下文
    async with aio.insecure_channel(f"{args.host}:{args.port}") as channel:
        stub = cosyvoice_pb2_grpc.CosyVoiceStub(channel)
        request = cosyvoice_pb2.Request()
        request.stream = args.stream
        request.speed = args.speed
        request.text_frontend = args.text_frontend

        # 请求构造逻辑保持不变
        if args.mode == "sft":
            logging.info("Constructing sft request")
            sft_request = request.sft_request
            sft_request.tts_text = args.tts_text
            sft_request.spk_id = args.spk_id
        elif args.mode == "zero_shot":
            logging.info("Constructing zero_shot request")
            zero_shot_request = request.zero_shot_request
            zero_shot_request.tts_text = args.tts_text
            zero_shot_request.prompt_text = args.prompt_text
            prompt_speech = load_wav(args.prompt_wav, 16000)
            print(prompt_speech)
            zero_shot_request.prompt_audio = convert_audio_tensor_to_bytes(
                prompt_speech
            )
        elif args.mode == "cross_lingual":
            logging.info("Constructing cross_lingual request")
            cross_lingual_request = request.cross_lingual_request
            cross_lingual_request.tts_text = args.tts_text
            prompt_speech = load_wav(args.prompt_wav, 16000)
            cross_lingual_request.prompt_audio = convert_audio_tensor_to_bytes(
                prompt_speech
            )
        elif args.mode == "instruct2":
            logging.info("Constructing instruct request")
            instruct2_request = request.instruct2_request
            instruct2_request.tts_text = args.tts_text
            instruct2_request.instruct_text = args.instruct_text
            prompt_speech = load_wav(args.prompt_wav, 16000)
            instruct2_request.prompt_audio = convert_audio_tensor_to_bytes(
                prompt_speech
            )
        elif args.mode == "instruct2_by_spk_id":
            logging.info("Constructing instruct2_by_spk_id request")
            instruct2_by_spk_id_request = request.instruct2_by_spk_id_request
            instruct2_by_spk_id_request.tts_text = args.tts_text
            instruct2_by_spk_id_request.instruct_text = args.instruct_text
            instruct2_by_spk_id_request.spk_id = args.spk_id
        else:
            logging.info("Constructing zero_shot_by_spk_id request")
            zero_shot_by_spk_id_request = request.zero_shot_by_spk_id_request
            zero_shot_by_spk_id_request.tts_text = args.tts_text
            zero_shot_by_spk_id_request.spk_id = args.spk_id
        # 异步流式接收响应
        try:
            start_time = time.time()
            last_time = start_time
            chunk_index = 0
            tts_audio = b""
            response_stream = stub.Inference(request)
            async for response in response_stream:
                if response.tts_audio:
                    tts_audio += response.tts_audio
                    logging.info(
                        f"Received audio chunk {len(response.tts_audio)} bytes, chunk index: {chunk_index},"
                        + f"cost {time.time() - last_time:.3f}s  all cost {time.time() - start_time:.3f}s "
                    )
                    last_time = time.time()
                    chunk_index += 1

            # 音频后处理
            if tts_audio:
                tts_speech = convert_audio_bytes_to_tensor(tts_audio)
                torchaudio.save(args.tts_wav, tts_speech, args.target_sr)
                logging.info(
                    f"Audio saved to {args.tts_wav} (duration: {tts_speech.shape[1] / args.target_sr:.2f}s) "
                    + f"cost {time.time() - last_time:.3f}s  all cost {time.time() - start_time:.3f}s "
                )
            else:
                logging.warning("Received empty audio response")

        except grpc.RpcError as e:
            logging.error(f"RPC error occurred: {e.code()}: {e.details()}")
        except Exception as e:
            logging.error(f"Processing failed: {str(e)}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument(
        "--mode",
        default="zero_shot_by_spk_id",
        choices=[
            "sft",
            "zero_shot",
            "cross_lingual",
            "instruct2",
            "instruct2_by_spk_id",
            "zero_shot_by_spk_id",
        ],
        help="Request mode",
    )
    parser.add_argument(
        "--stream", action="store_true", help="Streaming inference mode"
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Speed up the audio")
    parser.add_argument(
        "--text_frontend", type=bool, default=True, help="Text frontend mode"
    )
    parser.add_argument(
        "--tts_text",
        type=str,
        default="你好，我是通义千问语音合成大模型，请问有什么可以帮您的吗？",
    )
    parser.add_argument("--spk_id", type=str, default="001")
    parser.add_argument(
        "--prompt_text", type=str, default="希望你以后能够做的比我还好呦。"
    )
    parser.add_argument(
        "--prompt_wav", type=str, default="../../../asset/zero_shot_prompt.wav"
    )
    parser.add_argument(
        "--instruct_text",
        type=str,
        default="Theo 'Crimson', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.",
    )
    parser.add_argument("--tts_wav", type=str, default="demo.wav")
    parser.add_argument(
        "--target_sr",
        type=int,
        default=24000,
        help="Target sample rate for output audio, cosyvoice2 is 24000",
    )
    args = parser.parse_args()

    # 运行异步主函数
    asyncio.run(main(args))
