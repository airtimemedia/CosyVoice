import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio


def cosyvoice_example():
    """ CosyVoice Usage, check https://fun-audio-llm.github.io/ for more details
    """
    cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice-300M-SFT')
    # sft usage
    print(cosyvoice.list_available_spks())
    # change stream=True for chunk stream inference
    for i, j in enumerate(cosyvoice.inference_sft('你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？', '中文女', stream=False)):
        torchaudio.save('sft_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice-300M')
    # zero_shot usage
    for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', './asset/zero_shot_prompt.wav')):
        torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    # cross_lingual usage, <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
    for i, j in enumerate(cosyvoice.inference_cross_lingual('<|en|>And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that\'s coming into the family is a reason why sometimes we don\'t buy the whole thing.',
                                                            './asset/cross_lingual_prompt.wav')):
        torchaudio.save('cross_lingual_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    # vc usage
    for i, j in enumerate(cosyvoice.inference_vc('./asset/cross_lingual_prompt.wav', './asset/zero_shot_prompt.wav')):
        torchaudio.save('vc_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice-300M-Instruct')
    # instruct usage, support <laughter></laughter><strong></strong>[laughter][breath]
    for i, j in enumerate(cosyvoice.inference_instruct('在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。', '中文男',
                                                       'Theo \'Crimson\', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.<|endofprompt|>')):
        torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)


def cosyvoice2_example():
    """ CosyVoice2 Usage, check https://funaudiollm.github.io/cosyvoice2/ for more details
    """
    cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B')

    # NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
    # zero_shot usage
    for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', './asset/zero_shot_prompt.wav')):
        torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # save zero_shot spk for future usage
    assert cosyvoice.add_zero_shot_spk('希望你以后能够做的比我还好呦。', './asset/zero_shot_prompt.wav', 'my_zero_shot_spk') is True
    for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '', '', zero_shot_spk_id='my_zero_shot_spk')):
        torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    cosyvoice.save_spkinfo()

    # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
    for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', './asset/zero_shot_prompt.wav')):
        torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # instruct usage
    for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话<|endofprompt|>', './asset/zero_shot_prompt.wav')):
        torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # bistream usage, you can use generator as input, this is useful when using text llm model as input
    # NOTE you should still have some basic sentence split logic because llm can not handle arbitrary sentence length
    def text_generator():
        yield '收到好友从远方寄来的生日礼物，'
        yield '那份意外的惊喜与深深的祝福'
        yield '让我心中充满了甜蜜的快乐，'
        yield '笑容如花儿般绽放。'
    for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '希望你以后能够做的比我还好呦。', './asset/zero_shot_prompt.wav', stream=False)):
        torchaudio.save('zero_shot_bistream_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)


def cosyvoice3_example():
    """ CosyVoice3 Usage, check https://funaudiollm.github.io/cosyvoice3/ for more details
    """
    cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
    # zero_shot usage
    
    
    prompt_texts = [
        "and tie the demand of my horse. Emperor of China, Emperor of Mongolia, I will be Emperor of the World!"
        # "موسیقی",
        # "こんにちは!素晴らしい一日を!仕事の仕事に興味深いプロジェクトを終わってきました。そして今、お昼に何を作っているか考えています。ラーメンか、もしかしたら寿司かもしれません。あ、あと、明年京都に旅行を行っています。テリブラサームを見るのが待ちきれないです!",
        # "こんにちは!素晴らしい一日を!本を終わって、夕食の作り方を考えています。ラーメンも食べてみようかな。もしかしたら、今からカレーの新しいレシピを試してみようかな。決めた決めた。"
        # "お、こんにちは!私もとても嬉しいです。良い一日をお過ごしください。カフェで仕事を終わって、今、パークに行くのを考えています。本当に素晴らしい一日ですね。もしかしたら、マッチャラテを食べるかもしれません。後に何か計画がありますか?"
        # "एक रेसिपी दो बार बनानी चाहिए, नहीं बनानी चाहिए न, जिन्दगी चोटी है, वही रेसिपी बार बार बार बार बार बनाए जाओ तो फिर मतलब क्या फायदा, मेरे को इसलिए बारी बारी रेस्टोरंट में काम करने से डर लगता है, वही रेसिपी रोज ठोको रोज ठोको, लेकिन एक रेसिपी ऐसी है, जो मुझे मेरी टीम ने सोचने पर मजबूर कर दिया, कि शब यू ए रेसिपी दो बारा बनाओ, क्योंकि आपने जो बनाई है, वो दाल मजबूर"
    ]
    
    prompt_audios = [
        "/root/.local/share/References/selected-samples_trimmed_24s/13b63589-7a68-414d-8f3f-093747d91a6f.wav"
        # "/root/.local/share/References/selected-samples_trimmed_24s/07868e6d-fb36-4cb7-b2a4-b34dfa9ecb13.wav",
        # "/root/.local/share/References/11_labs_p2v_trimmed_24s/730a700b-5429-485c-a74f-1e8577cfb669.wav",
        # "/root/.local/share/References/11_labs_p2v_trimmed_24s/94f6a8a0-87a3-474c-89d8-bc1fdeab9155.wav",
        # "/root/.local/share/References/11_labs_p2v_trimmed_24s/dc17db59-77de-462a-89d9-690bf992ea63.wav",
        # "/root/.local/share/References/selected-samples_trimmed_24s/62e57fdd-5455-4fd2-8e0e-2295afdea0b9.wav"
    ]
    
    target_texts = [
        "send yours first"
        # "show me we are the world!",
        # "Hello! What a wonderful day! I just finished an interesting project for work, and now I'm thinking about what to make for lunch. Maybe some ramen, or perhaps sushi. Oh, and I'm also planning a trip to Kyoto next year. I can't wait to see the terriblasam!",
        # "Hello! What a wonderful day! I just finished a book and now I'm thinking about what to make for dinner. Maybe I'll try some ramen. Or perhaps I'll try a new curry recipe now. I've decided, I've decided.",
        # "Oh, hello! I'm also very happy. Have a great day. I just finished some work at a cafe and now I'm thinking about going to the park. It's really a wonderful day",
        # "maybe a recipe should be made twice, not made, life is short, if the same recipe is made again and again then what is the use, that's why I am afraid of working in restaurants one by one, the same recipe every day"
    ]
    
    for index, (prompt_text, prompt_audio, target_text) in enumerate(zip(prompt_texts, prompt_audios, target_texts)):
        print(f"Processing prompt_text: {prompt_text}, prompt_audio: {prompt_audio}, target_text: {target_text}")
        for i, j in enumerate(cosyvoice.inference_zero_shot(target_text, f'You are a helpful assistant.<|endofprompt|>{prompt_text}',
                                                            prompt_audio, stream=False)):
            torchaudio.save(f'zero_shot_{index}_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)

    # # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L280
    # for i, j in enumerate(cosyvoice.inference_cross_lingual('You are a helpful assistant.<|endofprompt|>[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，都很熟悉。[breath]',
    #                                                         './asset/zero_shot_prompt.wav', stream=False)):
    #     torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # # instruct usage, for supported control, check cosyvoice/utils/common.py#L28
    # for i, j in enumerate(cosyvoice.inference_instruct2('好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。', 'You are a helpful assistant. 请用广东话表达。<|endofprompt|>',
    #                                                     './asset/zero_shot_prompt.wav', stream=False)):
    #     torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    # for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', 'You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>',
    #                                                     './asset/zero_shot_prompt.wav', stream=False)):
    #     torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # # hotfix usage
    # for i, j in enumerate(cosyvoice.inference_zero_shot('高管也通过电话、短信、微信等方式对报道[j][ǐ]予好评。', 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
    #                                                     './asset/zero_shot_prompt.wav', stream=False)):
    #     torchaudio.save('hotfix_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)


def main():
    # cosyvoice_example()
    # cosyvoice2_example()
    cosyvoice3_example()


if __name__ == '__main__':
    main()
