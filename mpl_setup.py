import matplotlib

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}\usepackage{amsmath}'
matplotlib.rcParams['font.family'] = 'lmodern'
matplotlib.rcParams['font.size'] = 9

matplotlib.rcParams['figure.constrained_layout.use'] = True
matplotlib.rcParams['figure.constrained_layout.h_pad'] = 0.01
matplotlib.rcParams['figure.constrained_layout.w_pad'] = 0.01
matplotlib.rcParams['figure.figsize'] = [3.4, 2.27]

matplotlib.rcParams['savefig.bbox'] = 'standard'
matplotlib.rcParams['savefig.transparent'] = True

matplotlib.rcParams['lines.dashed_pattern'] = [3, 3]
matplotlib.rcParams['lines.dotted_pattern'] = [1, 2]
matplotlib.rcParams['lines.linewidth'] = 1.2
matplotlib.rcParams['axes.linewidth'] = .75

matplotlib.rcParams['legend.frameon'] = False
matplotlib.rcParams['legend.borderpad'] = 0.25
matplotlib.rcParams['legend.labelspacing'] = 0.25
matplotlib.rcParams['legend.handlelength'] = 1.65

# https://www.nature.com/articles/nmeth.1618
ORANGE = (230 / 255, 159 / 255, 0)
SKY_BLUE = (86 / 255, 180 / 255, 233 / 255)
YELLOW = (240 / 255, 228 / 255, 66 / 255)
BLUE = (0, 114 / 255, 178 / 255)
VERMILLION = (213 / 255, 94 / 255, 0)
GRAY = (0.3, 0.3, 0.3)