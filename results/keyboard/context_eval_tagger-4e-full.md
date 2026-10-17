# Context vs standalone word conversion: models/onnx/tagger-4e-full

## chat_test: 30 aligned messages, 122 words (0s)

- word accuracy alone: **82.0%**
- word accuracy with 2-word context: **86.1%**
- both wrong: 13.1%

Fixed by context:
- test server [dr] → alone 'дэр', context 'дээр'
- hiichlee, shalgaad [uz] → alone 'үзз', context 'үз'
- dahiad [uz] → alone 'үзз', context 'үз'
- chi garsan [uu] → alone 'өу', context 'уу'
- garna, zogsool [dr] → alone 'дэр', context 'дээр'
- tgvl lg [uz,] → alone 'үзз', context 'үз'

Broken by context:
- chi hiichih, [bi] → alone 'би', context 'бии'

## real_test: 109 aligned messages, 800 words (1s)

- word accuracy alone: **83.6%**
- word accuracy with 2-word context: **87.0%**
- both wrong: 11.8%

Fixed by context:
- amidrah yum [uu] → alone 'өу', context 'уу'
- emegtei yum [uu?] → alone 'өу', context 'уу'
- bidniigee baysgaad [bj] → alone 'байй', context 'байж'
- l bh [like] → alone 'лике', context 'like'
- best songsiin [hotlogch] → alone 'хөтлөгчих', context 'хөтлөгч'
- hotlogch nart [oshoo] → alone 'ошоө', context 'өшөө'
- aytai bish [ch] → alone 'чх', context 'ч'
- bgaa n [like] → alone 'лике', context 'like'

Broken by context:
- az jargal [husey] → alone 'хүсье', context 'хүсэе'
- huuchnii duug [hha] → alone 'хаха', context 'хха'
- bj bgaadaa [hhe] → alone 'хэхэ', context 'хххэ'
- n like [haha] → alone 'хаха', context 'хааха'
- Haayaa [muu] → alone 'муу', context 'а муу'
- hutsaad bga [Al] → alone 'ал', context 'ль'
- 2023 [hurtl] → alone 'хүртэл', context 'хуртэл'
- Huurhun [duu] → alone 'дуу', context 'дөө'
