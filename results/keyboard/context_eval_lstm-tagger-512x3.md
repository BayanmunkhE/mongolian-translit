# Context vs standalone word conversion: models/onnx/lstm-tagger-512x3

## chat_test: 30 aligned messages, 122 words (0s)

- word accuracy alone: **91.8%**
- word accuracy with 2-word context: **91.8%**
- both wrong: 4.9%

Fixed by context:
- hiichih, bi [test] → alone 'тeст', context 'test'
- test server [dr] → alone 'dр', context 'дээр'
- aa env [var] → alone 'вар', context 'var'
- garna, zogsool [dr] → alone 'dр', context 'дээр'

Broken by context:
- za tgvl [deploy-g] → alone 'deploy-г', context 'deploy-гаг'
- chi hiichih, [bi] → alone 'би', context 'бби'
- aa [env] → alone 'env', context 'эннү'
- za [5d] → alone '5-д', context '5д'

## real_test: 109 aligned messages, 800 words (1s)

- word accuracy alone: **89.9%**
- word accuracy with 2-word context: **89.6%**
- both wrong: 8.9%

Fixed by context:
- bna universe [best] → alone 'бэст', context 'best'
- miharu ter [chini] → alone 'чиньй', context 'чинь'
- uzchihsen baisan [TV] → alone 'төв', context 'тв'
- stoi hoorhon [love] → alone 'loвe', context 'love'
- hoorhon love [girls] → alone 'гиллс', context 'girls'
- ued goy [bje❤] → alone 'жээ❤', context 'байжээ❤'
- zaluu mun [uu] → alone 'уу', context 'үү'
- sonsogch n [bolchloo] → alone 'больлло', context 'болчлоо'

Broken by context:
- udsan bn [huuchnii] → alone 'хуучны', context 'ууччны'
- chini em [sh] → alone 'ш', context 'шү'
- duu shvv [❤❤] → alone '❤❤', context '❤❤❤'
- hun bn [hha] → alone 'хаха', context 'хаа'
- Huurhun [duu] → alone 'дуу', context 'дөө'
- ym baina [sh] → alone 'ш', context 'үү'
- yooy [yg] → alone 'яг', context 'үг'
- hoviy hh😊 [bi] → alone 'би', context 'бий'
