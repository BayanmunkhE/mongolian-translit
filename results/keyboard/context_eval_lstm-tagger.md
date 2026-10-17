# Context vs standalone word conversion: models/onnx/lstm-tagger

## chat_test: 30 aligned messages, 122 words (0s)

- word accuracy alone: **91.0%**
- word accuracy with 2-word context: **89.3%**
- both wrong: 8.2%

Fixed by context:
- chi garsan [uu] → alone 'өу', context 'уу'

Broken by context:
- chi hiichih, [bi] → alone 'би', context 'бий'
- za bolloo, [bayrlalaa] → alone 'баярлалаа', context 'байрлалаа'
- bsn, odoo [mdku] → alone 'мэдэхгүй', context 'мэдэдгүй'

## real_test: 109 aligned messages, 800 words (1s)

- word accuracy alone: **87.6%**
- word accuracy with 2-word context: **88.4%**
- both wrong: 10.6%

Fixed by context:
- amidrah yum [uu] → alone 'өу', context 'уу'
- duu bna [talagdlaaaaa] → alone 'таллагдла', context 'таалагдлаа'
- boliochee sanaa [zowmoor] → alone 'зөвмоор', context 'зовмоор'
- zaluusaaa amjilt [:D] → alone 'д', context 'd'
- ued goy [bje❤] → alone 'байээ❤', context 'байжээ❤'
- Tarchihsan yum [uu] → alone 'өу', context 'уу'
- zaluu mun [uu] → alone 'өу', context 'үү'
- ni bur [aimr] → alone 'аммар', context 'аймар'

Broken by context:
- huuchnii duug [hha] → alone 'хаха', context 'ха'
- nart oshoo [goy] → alone 'гоё', context 'ө гоё'
- jiliin daraa [dahij] → alone 'дахиж', context 'дахьж'
- huuyee [goy] → alone 'гоё', context 'э гоё'
- hun bn [hha] → alone 'хаха', context 'аха'
- yooy [yg] → alone 'яг', context 'үг'
- hoviy hh😊 [bi] → alone 'би', context 'бий'
- sonsoj bui [bi:🤣🤣] → alone 'би:🤣🤣', context 'бий:🤣🤣'
