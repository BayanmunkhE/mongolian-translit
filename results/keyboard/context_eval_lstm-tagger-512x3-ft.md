# Context vs standalone word conversion: models/onnx/lstm-tagger-512x3-ft

## chat_test: 30 aligned messages, 122 words (0s)

- word accuracy alone: **91.0%**
- word accuracy with 2-word context: **91.8%**
- both wrong: 5.7%

Fixed by context:
- test server [dr] → alone 'др', context 'дээр'
- aa env [var] → alone 'вар', context 'var'
- garna, zogsool [dr] → alone 'др', context 'дээр'
- chi [ch] → alone 'чи', context 'ч'

Broken by context:
- za tgvl [deploy-g] → alone 'deploy-г', context 'deploy-гэа'
- aa [env] → alone 'env', context 'эннү'
- za [5d] → alone '5-д', context '5д'

## real_test: 109 aligned messages, 800 words (1s)

- word accuracy alone: **89.4%**
- word accuracy with 2-word context: **90.2%**
- both wrong: 8.4%

Fixed by context:
- hun uzdg [biljee] → alone 'билжээ', context 'болжээ'
- bna universe [best] → alone 'бэст', context 'best'
- aytai bish [ch] → alone 'чи', context 'ч'
- chini em [sh] → alone 'үү', context 'ш'
- alna geh [ch] → alone 'чи', context 'ч'
- stoi hoorhon [love] → alone 'loвe', context 'love'
- hoorhon love [girls] → alone 'грллс', context 'girls'
- zaluusaaa amjilt [:D] → alone '-д', context 'd'

Broken by context:
- Mongolchuudaa [harhad] → alone 'харахад', context 'хархад'
- Ene [Onorzul] → alone 'өнөрзул', context 'онөрзул'
- udsan bn [huuchnii] → alone 'хуучны', context 'ууччны'
- duu shvv [❤❤] → alone '❤❤', context '❤❤❤'
- hun bn [hha] → alone 'хаха', context 'ааа'
- Huurhun [duu] → alone 'дуу', context 'дөө'
- yooy [yg] → alone 'яг', context 'үг'
- hoviy hh😊 [bi] → alone 'би', context 'бий'
