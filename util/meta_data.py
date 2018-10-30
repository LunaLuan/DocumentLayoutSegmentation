LIST_OF_NONTEXT_REGION = ['ImageRegion', 'GraphicRegion', 'ChartRegion', 'LineDrawingRegion',
                  'SeparatorRegion', 'TableRegion', 'MathsRegion', 'ChemRegion', 'MusicRegion',
                  'AdvertRegion', 'NoiseRegion', 'UnknownRegion', 'FrameRegion']
LIST_OF_NONTEXT_COLOR = [(0,0,150), (0,0,255), (0,150,0), (0,150,150),
                         (0,150,255), (0,255,0), (0,255,150), (0,255,255), (150,0,0),
                         (150,0,150), (150,0,255), (150,150,0), (150,150,150)] #RGB

LIST_OF_TEXT_TYPE = ['heading', 'header', 'paragraph', 'drop-capital',
                     'caption', 'floating', 'credit', 'footer',
                     'page-number', 'quote', 'sentence']
LIST_OF_TEXT_COLOR = [(150,150,255), (150,255,0), (150,255,150), (150,255,255),
                      (255,0,0), (255,0,150), (255,0,255), (255,150,0),
                      (255,150,150), (255,150,255), (255,255,0)]

REGION_COLOR = {}
COLOR_REGION = {}

for i in range(len(LIST_OF_NONTEXT_REGION)):
    region = LIST_OF_NONTEXT_REGION[i]
    color = LIST_OF_NONTEXT_COLOR[i]

    REGION_COLOR[region] = color
    COLOR_REGION[color] = region

for i in range(len(LIST_OF_TEXT_TYPE)):
    region = 'TextRegion' + ' ' + LIST_OF_TEXT_TYPE[i]
    color = LIST_OF_TEXT_COLOR[i]

    REGION_COLOR[region] = color
    COLOR_REGION[color] = region

