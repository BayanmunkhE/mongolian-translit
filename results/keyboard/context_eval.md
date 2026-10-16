# Context vs standalone word conversion: models/byt5-pruned-4e2d-1024-ft2

## chat_test: 30 aligned messages, 122 words (15s)

- word accuracy alone: **75.4%**
- word accuracy with 2-word context: **86.9%**
- both wrong: 10.7%

Fixed by context:
- bna [bna] → alone 'байна байна', context 'байна'
- bh udur [ni] → alone 'би нь', context 'нь'
- tgvl deploy-g [chi] → alone 'чи хийе', context 'чи'
- bi test [server] → alone 'сервер', context 'server'
- test server [dr] → alone 'дээр дээрэ', context 'дээр'
- server dr [bna] → alone 'байна байна', context 'байна'
- ok ok [10] → alone '10-н0', context '10'
- 500 ogood [bna] → alone 'байна байна', context 'байна'

Broken by context:
- bna bna [yu] → alone 'юу', context 'уу'
- za bolloo, [bayrlalaa] → alone 'баярлалаа', context 'байрлалаа'
- bsn, odoo [mdku] → alone 'мэдэхгүй', context 'мэдэхүү'

## real_test: 109 aligned messages, 800 words (132s)

- word accuracy alone: **74.2%**
- word accuracy with 2-word context: **87.2%**
- both wrong: 11.8%

Fixed by context:
- amidrah yum [uu] → alone 'үү', context 'уу'
- yum uu [chi] → alone 'чи хийе', context 'чи'
- goy duu [bna] → alone 'байна байна', context 'байна'
- duu bna [<3] → alone 'яу', context '<3'
- emegtei yum [uu?] → alone 'өө', context 'уу'
- zndoo udsan [bn] → alone 'байна байн', context 'байна'
- Mongol xaachix [ve,] → alone 'үе', context 'вэ'
- zav gargaad [l] → alone 'ли л', context 'л'

Broken by context:
- bj bgaadaa [hhe] → alone 'хэхэ', context 'хэзээ'
- jinkneesee sezontoi [boltson] → alone 'болцон', context 'болтсон'
- esto zvun [duu] → alone 'дуу', context 'дөө'
- Huurhun [duu] → alone 'дуу', context 'дөө'
- Mongol [pop] → alone 'поп', context 'pop'
- yawsan bol [yun] → alone 'юун', context 'юм'
- hmataa hoviy [hh😊] → alone 'хаха😊', context 'хийх😊'
- 1:17 deer [bataa] → alone 'батаа', context 'байтаа'
