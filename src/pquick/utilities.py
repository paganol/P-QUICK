from __future__ import annotations

import os
import re
import socket
from pathlib import Path

# Detector sets: base detector groups by frequency, then derived aliases below.
DETSETS: dict[str, tuple[str, ...]] = {
    # 100 GHz
    "100ds1": ("100-1a", "100-1b", "100-4a", "100-4b"),
    "100ds2": ("100-2a", "100-2b", "100-3a", "100-3b"),
    # 143 GHz
    "143ds1": ("143-1a", "143-1b", "143-3a", "143-3b"),
    "143ds2": ("143-2a", "143-2b", "143-4a", "143-4b"),
    "143swb": ("143-5", "143-6", "143-7"),
    # 217 GHz
    "217ds1": ("217-5a", "217-5b", "217-7a", "217-7b"),
    "217ds2": ("217-6a", "217-6b", "217-8a", "217-8b"),
    "217swb": ("217-1", "217-2", "217-3", "217-4"),
    # 353 GHz
    "353ds1": ("353-3a", "353-3b", "353-5a", "353-5b"),
    "353ds2": ("353-4a", "353-4b", "353-6a", "353-6b"),
    "353swb": ("353-1", "353-2", "353-7", "353-8"),
    # 545 GHz
    "545dsA": ("545-1",),
    "545dsB": ("545-2", "545-4"),
    "545ghz": ("545-1", "545-2", "545-4"),
    # 857 GHz
    "857dsA": ("857-1", "857-3"),
    "857dsB": ("857-2", "857-4"),
    "857ghz": ("857-1", "857-2", "857-3", "857-4"),
}

# Derived aliases and channel-wide groupings.
DETSETS["100dsA"] = DETSETS["100ds1"]
DETSETS["100dsB"] = DETSETS["100ds2"]
DETSETS["100psb"] = DETSETS["100ds1"] + DETSETS["100ds2"]
DETSETS["100ghz"] = DETSETS["100psb"]

DETSETS["143dsA"] = DETSETS["143ds1"] + ("143-5", "143-7")
DETSETS["143dsB"] = DETSETS["143ds2"] + ("143-6",)
DETSETS["143psb"] = DETSETS["143ds1"] + DETSETS["143ds2"]
DETSETS["143ghz"] = DETSETS["143psb"] + DETSETS["143swb"]

DETSETS["217dsA"] = DETSETS["217ds1"] + ("217-5", "217-7")
DETSETS["217dsB"] = DETSETS["217ds2"] + ("217-6", "217-8")
DETSETS["217psb"] = DETSETS["217ds1"] + DETSETS["217ds2"]
DETSETS["217ghz"] = DETSETS["217psb"] + DETSETS["217swb"]

DETSETS["353dsA"] = DETSETS["353ds1"] + ("353-1", "353-7")
DETSETS["353dsB"] = DETSETS["353ds2"] + ("353-2", "353-8")
DETSETS["353psb"] = DETSETS["353ds1"] + DETSETS["353ds2"]
DETSETS["353ghz"] = DETSETS["353psb"] + DETSETS["353swb"]

# Per-release focal-plane geometry and polarisation efficiency, extracted from
# the RIMO FITS files. Per detector:
#   (phi_uv_deg, theta_uv_deg, psi_uv_deg, psi_pol_deg, epsilon)
# P-QUICK convention: the beam ellipse rotates with psi_uv; psi_pol is an extra
# rotation of the polarisation axis only (map-making). Both tables are stored in
# this convention.
#
# NPIPE/PR4: npipe-symmetrized RIMOs (HFI_RIMO_R4.00 is bit-identical).
FOCAL_PLANE_NPIPE: dict[str, tuple[float, float, float, float, float]] = {
    # LFI (RIMO_LFI_npipe5_symmetrized.fits)
    "LFI18S": (-131.828280713, 3.334115892, 22.15, 0.0, 0.0),
    "LFI18M": (-131.828280713, 3.334115892, 22.15, 90.2, 0.0),
    "LFI19S": (-150.481524354, 3.2094237, 22.4, 0.0, 0.0),
    "LFI19M": (-150.481524354, 3.2094237, 22.4, 90.0, 0.0),
    "LFI20S": (-168.182364352, 3.184449966, 22.38, 0.0, 0.0),
    "LFI20M": (-168.182364352, 3.184449966, 22.38, 89.9, 0.0),
    "LFI21S": (169.280933171, 3.185856133, -22.38, 0.0, 0.0),
    "LFI21M": (169.280933171, 3.185856133, -22.38, 90.1, 0.0),
    "LFI22S": (151.360376597, 3.173745645, -22.34, 0.1, 0.0),
    "LFI22M": (151.360376597, 3.173745645, -22.34, 90.1, 0.0),
    "LFI23S": (132.259118023, 3.281368261, -22.08, 0.0, 0.0),
    "LFI23M": (132.259118023, 3.281368261, -22.08, 89.7, 0.0),
    "LFI24S": (-179.539804815, 4.07251102, 0.01, 0.0, 0.0),
    "LFI24M": (-179.539804815, 4.07251102, 0.01, 90.0, 0.0),
    "LFI25S": (61.092970732, 4.983509821, -113.23, 0.0, 0.0),
    "LFI25M": (61.092970732, 4.983509821, -113.23, 89.5, 0.0),
    "LFI26S": (-61.670013546, 5.035819216, 113.23, 0.0, 0.0),
    "LFI26M": (-61.670013546, 5.035819216, 113.23, 90.5, 0.0),
    "LFI27S": (153.987200079, 4.345964487, -22.46, 0.0, 0.0),
    "LFI27M": (153.987200079, 4.345964487, -22.46, 89.7, 0.0),
    "LFI28S": (-153.423530136, 4.375677172, 22.45, 0.0, 0.0),
    "LFI28M": (-153.423530136, 4.375677172, 22.45, 90.3, 0.0),
    # HFI (RIMO_HFI_npipe5v16_symmetrized.fits)
    "100-1a": (-141.05935614227732, 1.9511712310837863, 23.100926005468104, 0.21718725535964634, 0.029089286784225586),
    "100-1b": (-141.05935614227732, 1.9511712310837863, -68.19903882146515, 0.1265552541580502, 0.015322789270269683),
    "100-2a": (-166.7949756487086, 1.8399252948998028, 45.101146328266395, -0.8362278127465876, 0.020255166888394183),
    "100-2b": (-166.7949756487086, 1.8399252948998028, -45.698894626708025, 0.26877767513361817, 0.054065869096409126),
    "100-3a": (168.9526859623529, 1.823443939697911, 0.20116097433238955, -0.8680171012398928, 0.05506067403457874),
    "100-3b": (168.9526859623529, 1.823443939697911, -89.89889318975888, 0.18568302854653407, 0.035056533893697166),
    "100-4a": (142.72719593261863, 1.90506322224582, -23.4990689064743, 0.46236422179839576, 0.03156121575218292),
    "100-4b": (142.72719593261863, 1.90506322224582, 68.00093375183148, -0.05543611883731446, 0.0361790125211971),
    "143-1a": (-49.94566096416487, 1.845262025323955, 45.499275429985175, 0.23588686174489032, 0.07786670237010133),
    "143-1b": (-49.94566096416487, 1.845262025323955, -42.30074605810816, -0.03922772048498853, 0.0727998907497113),
    "143-2a": (-26.444550122040102, 1.3569792755332817, 45.399274462946686, -0.21338579793364978, 0.07902907144226831),
    "143-2b": (-26.444550122040102, 1.3569792755332817, -44.80079187932733, 0.29247950677952195, 0.058368121021314355),
    "143-3a": (23.831852779621244, 1.3029465576984236, -0.30070614780866867, 0.3829919948337097, 0.07507674355793059),
    "143-3b": (23.831852779621244, 1.3029465576984236, -86.90072324559468, -0.35538131974003845, 0.05995956407295109),
    "143-4a": (49.203828792521875, 1.8594299776870584, 0.8993106149607464, -0.28860787164716456, 0.043037066702203024),
    "143-4b": (49.203828792521875, 1.8594299776870584, 89.29930172539522, 0.31212479988740827, 0.032411625182424325),
    "143-5": (-34.38099534349748, 2.094413100919524, 65.69904102677158, 3.248172592774446, 0.8601510734279166),
    "143-6": (-10.975189975562275, 1.7886973045279844, 70.59901240937909, -0.8932870004123626, 0.9254844835576308),
    "143-7": (8.616446035693382, 1.7513075823762128, 102.79902422135243, -7.561532341738046, 0.9406667894652256),
    "143-8": (0.5830130273149843, 0.03671177306686759, 0.0, 0.0, 0.978),
    "217-1": (-134.42876517865145, 1.446373718753991, 98.40051608242797, -1.2937632020658736, 0.9543178934524726),
    "217-2": (-160.59762275316263, 1.0457517508853698, 82.50050222241266, 13.365904858774682, 0.9355507648786278),
    "217-3": (164.1507716715763, 1.050744508908937, 170.90051360691288, -187.8815897571469, 0.9756221726047231),
    "217-4": (135.42542955270926, 1.3850080448369897, 120.00054809009784, 1.2001506643440516, 0.9272203732322306),
    "217-5a": (-111.9506998440139, 1.3606216433860534, 47.00030546165768, 0.0804572725331857, 0.01966613955496865),
    "217-5b": (-111.9506998440139, 1.3606216433860534, -43.89973557083826, -0.22061205127023462, 0.013451323830300722),
    "217-6a": (-129.86658628215415, 0.7517541126553046, 46.10028116669726, -0.09356301734091667, 0.05198948355031017),
    "217-6b": (-129.86658628215415, 0.7517541126553046, -44.09974979921443, 0.06541010566214372, 0.03140935922693773),
    "217-7a": (134.4584054942768, 0.7223693841821932, -0.3997018363502834, -0.16785763693942152, 0.03047345667068867),
    "217-7b": (134.4584054942768, 0.7223693841821932, -89.49971153036488, 0.03309635201362196, 0.024845549683928145),
    "217-8a": (111.81000965956375, 1.2923431056918357, 0.3002962157060683, 0.8792591411945343, 0.01647171555956994),
    "217-8b": (111.81000965956375, 1.2923431056918357, -89.29971291314057, -0.8007424420873306, 0.021459292283995877),
    "353-1": (-90.06101082712358, 2.1010432854396073, 103.09996168080303, 358.64173270754844, 0.9953939449322853),
    "353-2": (-89.01210889923935, 1.4591541161488497, 114.59995033914085, 6.076985042553878, 0.8606254230834106),
    "353-3a": (-89.95519699347143, 0.8604640098401124, 45.500007713935915, 0.20316928005584395, 0.053029246129037345),
    "353-3b": (-89.95519699347143, 0.8604640098401124, -46.10003389386922, -0.2647513415198949, 0.04767290967012182),
    "353-4a": (-83.57358988096817, 0.2421442042030837, 45.70000294108355, -0.028413339206172193, 0.06749588478410687),
    "353-4b": (-83.57358988096817, 0.2421442042030837, -44.40003808648236, -0.08045280895962038, 0.05347648675285093),
    "353-5a": (89.3779941561846, 0.33825072108964144, -2.299997143090937, 0.17006123101458231, 0.07850005755134623),
    "353-5b": (89.3779941561846, 0.33825072108964144, 89.60000289647817, -0.16784708625667552, 0.04741434525761156),
    "353-6a": (88.29780089498226, 0.9360623604458883, -0.3000019690752889, 0.5187990255994472, 0.08112797342531354),
    "353-6b": (88.29780089498226, 0.9360623604458883, 89.50000064601421, -0.6391824347443676, 0.0386269962953014),
    "353-7": (89.89791796303953, 1.496847493106246, 121.50000737550748, 5.313433906375362, 0.8990996629825269),
    "353-8": (89.22553304057836, 2.0340444095731045, 132.99999611692417, 6.303584483025396, 0.8202234625834485),
    "545-1": (-76.27903472282918, 2.1779591406768963, 129.09964417861718, 0.0, 0.913),
    "545-2": (-69.6774034238106, 1.569514099698148, 139.09964577147232, 0.0, 0.894),
    "545-3": (70.9188114054119, 1.5776386382889418, 150.3, 0.0, 0.9),
    "545-4": (74.91602574246608, 2.1089476863039724, 145.59971129953055, 0.0, 0.89),
    "857-1": (-59.73943550173863, 1.0318692856366596, 157.29967008820407, 0.0, 0.887),
    "857-2": (-25.853224783435106, 0.608849000101285, 108.39966388508951, 0.0, 0.883),
    "857-3": (30.97131205110658, 0.6133117406071876, 176.79969876595098, 0.0, 0.856),
    "857-4": (59.59114170424986, 1.0809538990284482, 161.8996858089204, 0.0, 0.897),
}

# PR3: LFI_RIMO_R2.50 (same convention, taken verbatim) + HFI_RIMO_R2.00.
# HFI R2.00 stores the whole orientation in PSI_POL with PSI_UV = 0 (its value
# matches npipe's PSI_UV to float precision), so it is mapped into the P-QUICK
# convention as psi_uv := PSI_POL, psi_pol := 0 — faithful to PR3, which had no
# separate pol-axis fine offset. phi/theta and epsilon are R2.00's own values.
FOCAL_PLANE_PR3: dict[str, tuple[float, float, float, float, float]] = {
    # LFI (LFI_RIMO_R2.50.fits)
    "LFI18S": (-131.81993418, 3.334289196, 22.15, 0.0, 0.0011071335),
    "LFI18M": (-131.828280713, 3.334115892, 22.15, 90.2, 0.0016323001),
    "LFI19S": (-150.4877839, 3.20940304, 22.4, 0.0, 0.0011981191),
    "LFI19M": (-150.481524354, 3.2094237, 22.4, 90.0, 0.002538049),
    "LFI20S": (-168.194131988, 3.184805499, 22.38, 0.0, 0.00031739494),
    "LFI20M": (-168.182364352, 3.184449966, 22.38, 89.9, 0.00067904726),
    "LFI21S": (169.270800509, 3.185314648, -22.38, 0.0, 0.0002529298),
    "LFI21M": (169.280933171, 3.185856133, -22.38, 90.1, 0.00079744364),
    "LFI22S": (151.37056161, 3.173953785, -22.34, 0.1, 0.0014581426),
    "LFI22M": (151.360376597, 3.173745645, -22.34, 90.1, 0.0027720432),
    "LFI23S": (132.279661507, 3.281360837, -22.08, 0.0, 0.0017640058),
    "LFI23M": (132.259118023, 3.281368261, -22.08, 89.7, 0.0022913952),
    "LFI24S": (-179.505392209, 4.071417043, 0.01, 0.0, 0.00016791906),
    "LFI24M": (-179.539804815, 4.07251102, 0.01, 90.0, 0.00017322078),
    "LFI25S": (61.125309612, 4.983155894, -113.23, 0.0, 0.00079579279),
    "LFI25M": (61.092970732, 4.983509821, -113.23, 89.5, 0.00074954895),
    "LFI26S": (-61.675227539, 5.036198824, 113.23, 0.0, 0.0028333485),
    "LFI26M": (-61.670013546, 5.035819216, 113.23, 90.5, 0.0023561341),
    "LFI27S": (153.98498607, 4.346091361, -22.46, 0.0, 4.654789e-05),
    "LFI27M": (153.987200079, 4.345964487, -22.46, 89.7, 0.00013249516),
    "LFI28S": (-153.418020781, 4.375284092, 22.45, 0.0, 0.00010715193),
    "LFI28M": (-153.423530136, 4.375677172, 22.45, 90.3, 0.00010720129),
    # HFI (HFI_RIMO_R2.00.fits; orientation moved PSI_POL -> psi_uv)
    "100-1a": (-141.09277749011437, 1.953245551007349, 23.100926005480975, 0.0, 0.0272),
    "100-1b": (-141.08478330108278, 1.953362777608375, -68.19903882145002, 0.0, 0.0293),
    "100-2a": (-166.79549695452894, 1.8413920438415272, 45.10114632824032, 0.0, 0.0195),
    "100-2b": (-166.81707393176762, 1.8410708343448197, -45.69889462671103, 0.0, 0.0513),
    "100-3a": (168.96283310834986, 1.8248021461557542, 0.20116097433266553, 0.0, 0.0521),
    "100-3b": (168.934235822309, 1.8249803020684894, -89.89889318999445, 0.0, 0.0339),
    "100-4a": (142.77403388835836, 1.9064183272065576, -23.499068906452433, 0.0, 0.0219),
    "100-4b": (142.7413645214752, 1.907587423603999, 68.00093375181935, 0.0, 0.0402),
    "143-1a": (-49.96357835503376, 1.8452744611240457, 45.499275429996764, 0.0, 0.0915),
    "143-1b": (-49.93555920716649, 1.8444284428544386, -42.30074605810424, 0.0, 0.0835),
    "143-2a": (-26.47353828856772, 1.3565747209484624, 45.399274462949485, 0.0, 0.067),
    "143-2b": (-26.411067937376846, 1.3564022016309139, -44.80079187933428, 0.0, 0.0572),
    "143-3a": (23.83835863168934, 1.302252155081667, -0.30070614780861155, 0.0, 0.0875),
    "143-3b": (23.80159630392704, 1.3010108195822865, -86.90072324558632, 0.0, 0.053),
    "143-4a": (49.210589688997096, 1.8589379192547597, 0.8993106149597403, 0.0, 0.0357),
    "143-4b": (49.21094977965728, 1.8586185144474285, 89.2993017254333, 0.0, 0.0371),
    "143-5": (-34.397722906514225, 2.094329863663099, 65.69904102685662, 0.0, 0.882),
    "143-6": (-10.986092222893676, 1.7883316853685494, 70.59901240927847, 0.0, 0.92),
    "143-7": (8.602711722407957, 1.7510356217950256, 102.79902422154046, 0.0, 0.974),
    "143-8": (33.40418586629398, 2.1034296551735587, 75.70000000000002, 0.0, 0.978),
    "217-1": (-134.43990275470946, 1.4477415378850607, 98.40051608217783, 0.0, 0.926),
    "217-2": (-160.58987286689526, 1.0472597412076248, 82.50050222236695, 0.0, 0.961),
    "217-3": (164.1976344788386, 1.0517949447424713, 170.90051360677063, 0.0, 0.924),
    "217-4": (135.4776748672959, 1.3855375214190706, 120.0005480902742, 0.0, 0.916),
    "217-5a": (-111.96390062231833, 1.3619352344885043, 47.00030546164935, 0.0, 0.0256),
    "217-5b": (-111.96595625960072, 1.361811238939098, -43.89973557081458, 0.0, 0.0246),
    "217-6a": (-129.89502014283383, 0.7528010263066472, 46.100281166689776, 0.0, 0.026),
    "217-6b": (-129.8088726373262, 0.7542640494724631, -44.09974979920905, 0.0, 0.0236),
    "217-7a": (134.5464271081879, 0.7226869736504078, -0.3997018363504637, 0.0, 0.0307),
    "217-7b": (134.57186755513627, 0.7227789364378132, -89.49971153027391, 0.0, 0.0327),
    "217-8a": (111.8624826167968, 1.2920430927424065, 0.300296215706035, 0.0, 0.0298),
    "217-8b": (111.83939770049066, 1.2923997545364114, -89.29971291288545, 0.0, 0.0303),
    "353-1": (-90.09493249584187, 2.1017459289402916, 103.09996168084412, 0.0, 0.938),
    "353-2": (-89.04670188643921, 1.4599614482560657, 114.59995033890964, 0.0, 0.91),
    "353-3a": (-90.0272829115988, 0.8614109158078779, 45.50000771395072, 0.0, 0.0583),
    "353-3b": (-90.03908046950873, 0.860843671819801, -46.10003389385283, 0.0, 0.0413),
    "353-4a": (-83.79386519656116, 0.24260080963579544, 45.7000029410551, 0.0, 0.0683),
    "353-4b": (-83.91309672066559, 0.24613101109946667, -44.400038086464036, 0.0, 0.0439),
    "353-5a": (89.54309571816943, 0.3375268095920351, -2.2999971430917445, 0.0, 0.0829),
    "353-5b": (89.56387304223949, 0.33925139719172853, 89.60000289663135, 0.0, 0.0661),
    "353-6a": (88.33529722448932, 0.9357263559227187, -0.30000196907515453, 0.0, 0.0664),
    "353-6b": (88.32499554128862, 0.9370899808541753, 89.50000064576622, 0.0, 0.0595),
    "353-7": (89.93348909511062, 1.4962356961745724, 121.50000737576244, 0.0, 0.852),
    "353-8": (89.24336718827416, 2.033495050587014, 132.99999611672234, 0.0, 0.855),
    "545-1": (-76.3175464989974, 2.178835722965196, 129.0996441785806, 0.0, 0.913),
    "545-2": (-69.7140011646701, 1.5703669319274454, 139.0996457712249, 0.0, 0.894),
    "545-3": (71.2209994370323, 1.6244182057479344, 150.3, 0.0, 0.9),
    "545-4": (74.9472749922384, 2.1080774240532323, 145.59971129928914, 0.0, 0.89),
    "857-1": (-59.802278464399194, 1.033527936529367, 157.29967008813492, 0.0, 0.887),
    "857-2": (-26.105043877972886, 0.6074398683099843, 108.39966388496836, 0.0, 0.883),
    "857-3": (30.940120152421667, 0.6106390790103401, 176.79969876567986, 0.0, 0.856),
    "857-4": (59.60960056138405, 1.0792687341094405, 161.89968580874665, 0.0, 0.897),
}

_FOCAL_PLANES: dict[str, dict[str, tuple[float, float, float, float, float]]] = {
    "NPIPE": FOCAL_PLANE_NPIPE,
    "PR4": FOCAL_PLANE_NPIPE,  # PR4 == NPIPE (same data release)
    "PR3": FOCAL_PLANE_PR3,
}

# NPIPE per-detector map weights: both arms of a horn share the horn weight
# (ported from qp_planck/qp_planck/utilities.py detector_weights). Non-working
# bolometers (143-8, 545-3) are absent, so they get skipped.
NPIPE_DETECTOR_WEIGHTS: dict[str, float] = {
    "100-1a": 763430.0, "100-1b": 763430.0,
    "100-2a": 1266100.0, "100-2b": 1266100.0,
    "100-3a": 1063100.0, "100-3b": 1063100.0,
    "100-4a": 1053200.0, "100-4b": 1053200.0,
    "143-1a": 1640700.0, "143-1b": 1640700.0,
    "143-2a": 1857700.0, "143-2b": 1857700.0,
    "143-3a": 1643900.0, "143-3b": 1643900.0,
    "143-4a": 1445800.0, "143-4b": 1445800.0,
    "143-5": 2763000.0, "143-6": 2694200.0, "143-7": 2859900.0,
    "217-1": 1105800.0, "217-2": 1026100.0, "217-3": 1095800.0, "217-4": 1059300.0,
    "217-5a": 673180.0, "217-5b": 673180.0,
    "217-6a": 710920.0, "217-6b": 710920.0,
    "217-7a": 765760.0, "217-7b": 765760.0,
    "217-8a": 712260.0, "217-8b": 712260.0,
    "353-1": 128290.0, "353-2": 134750.0,
    "353-3a": 48067.0, "353-3b": 48067.0,
    "353-4a": 42187.0, "353-4b": 42187.0,
    "353-5a": 56914.0, "353-5b": 56914.0,
    "353-6a": 25293.0, "353-6b": 25293.0,
    "353-7": 87730.0, "353-8": 74453.0,
    "545-1": 4475.5, "545-2": 5540.3, "545-4": 4321.0,
    "857-1": 6.8895, "857-2": 6.3108, "857-3": 6.5964, "857-4": 3.6785,
    "LFI18M": 53650.0, "LFI18S": 53650.0,
    "LFI19M": 42141.0, "LFI19S": 42141.0,
    "LFI20M": 36579.0, "LFI20S": 36579.0,
    "LFI21M": 50355.0, "LFI21S": 50355.0,
    "LFI22M": 49363.0, "LFI22S": 49363.0,
    "LFI23M": 47966.0, "LFI23S": 47966.0,
    "LFI24M": 123720.0, "LFI24S": 123720.0,
    "LFI25M": 140490.0, "LFI25S": 140490.0,
    "LFI26M": 112330.0, "LFI26S": 112330.0,
    "LFI27M": 401640.0, "LFI27S": 401640.0,
    "LFI28M": 369000.0, "LFI28S": 369000.0,
}

MISSION_LENGTH_RANGES: dict[str, tuple[int, int]] = {
    "full": (91, 974),
    "hm1": (91, 563),
    "hm2": (564, 974),
    "survey1": (91, 270),
    "survey2": (271, 456),
    "survey3": (457, 636),
    "survey4": (637, 807),
    "survey5": (808, 974),
}


# PR3 per-detector map weights.
#  - HFI: SRoll per-detector (calib/NEP)^2 (DX11 calib / RD12 NEP) — differs between
#    a horn's a/b arms, unlike NPIPE.
#  - LFI: Planck-2018 (aa33293-18) per-horn Eq. 7 weight 2/(sigma_M^2 + sigma_S^2)
#    from the Table 4 white-noise levels [uK^2/Hz]; shared by a horn's M/S arms, e.g.
#    LFI27 = 2/(281.5 + 302.8). 143-8/545-3 absent (dead). Absolute scale is arbitrary
#    (it cancels in the per-pixel solve); only the within-channel ratios matter.
PR3_DETECTOR_WEIGHTS: dict[str, float] = {
    "100-1a": 162673.0, "100-1b": 227632.3,
    "100-2a": 674779.9, "100-2b": 346375.2,
    "100-3a": 903876.6, "100-3b": 610547.8,
    "100-4a": 416489.2, "100-4b": 226073.2,
    "143-1a": 1702989.0, "143-1b": 704902.4,
    "143-2a": 1740084.0, "143-2b": 1509531.0,
    "143-3a": 1435034.0, "143-3b": 1530800.0,
    "143-4a": 1276859.0, "143-4b": 1069561.0,
    "143-5": 2115344.0, "143-6": 2045240.0, "143-7": 2669092.0,
    "217-1": 1200040.0, "217-2": 1120057.0, "217-3": 1249481.0, "217-4": 1408729.0,
    "217-5a": 459494.2, "217-5b": 608457.2,
    "217-6a": 480378.1, "217-6b": 528185.9,
    "217-7a": 634943.2, "217-7b": 654692.2,
    "217-8a": 492636.1, "217-8b": 464691.9,
    "353-1": 166463.3, "353-2": 158481.0,
    "353-3a": 30471.63, "353-3b": 42703.47,
    "353-4a": 41951.33, "353-4b": 37735.2,
    "353-5a": 44020.63, "353-5b": 42320.42,
    "353-6a": 20896.82, "353-6b": 22874.54,
    "353-7": 109668.2, "353-8": 91233.83,
    "545-1": 2.572538, "545-2": 3.164496, "545-4": 2.7508,
    "857-1": 2.494976, "857-2": 2.561947, "857-3": 2.430096, "857-4": 1.28278,
    "LFI18M": 0.00204248, "LFI18S": 0.00204248,
    "LFI19M": 0.00176507, "LFI19S": 0.00176507,
    "LFI20M": 0.00165714, "LFI20S": 0.00165714,
    "LFI21M": 0.00197981, "LFI21S": 0.00197981,
    "LFI22M": 0.00195886, "LFI22S": 0.00195886,
    "LFI23M": 0.00191755, "LFI23S": 0.00191755,
    "LFI24M": 0.00231669, "LFI24S": 0.00231669,
    "LFI25M": 0.00246853, "LFI25S": 0.00246853,
    "LFI26M": 0.00220872, "LFI26S": 0.00220872,
    "LFI27M": 0.00342290, "LFI27S": 0.00342290,
    "LFI28M": 0.00331126, "LFI28S": 0.00331126,
}

_WEIGHT_SETS: dict[str, dict[str, float]] = {
    "NPIPE": NPIPE_DETECTOR_WEIGHTS,
    "PR4": NPIPE_DETECTOR_WEIGHTS,  # PR4 == NPIPE (same data release)
    "PR3": PR3_DETECTOR_WEIGHTS,
}


def detector_map_weight(detector: str, data_version: str = "NPIPE", default: float = 1.0) -> float:
    """Per-detector inverse-noise map weight for the chosen data release (``NPIPE``/``PR4``/``PR3``)."""
    return _WEIGHT_SETS[data_version.upper()].get(detector.strip(), default)


def has_detector_weight(detector: str, data_version: str = "NPIPE") -> bool:
    """True if the detector is in the chosen weight set, i.e. a working Planck detector.

    The set is the canonical good-detector list (cf qp_planck's ``list_planck(good=True)``):
    non-working bolometers — Planck HFI 143-8 and 545-3, the RTS-noise detectors — are
    deliberately absent.
    """
    return detector.strip() in _WEIGHT_SETS[data_version.upper()]


def is_psb(detector: str) -> bool:
    """True for a polarization-sensitive detector, False for an unpolarized SWB.

    Matches qp_planck: the name ends in ``a``/``b`` (HFI PSB arm) or ``M``/``S``
    (LFI radiometer arm); spider-web bolometers (e.g. ``143-5``) do not.
    """
    return detector.strip()[-1:] in "abMS"


def parse_mission_length(value: str) -> tuple[int, int]:
    """Parse a mission-length selector into an inclusive OD range.

    Supported values:
    - Named ranges: ``full``, ``survey1`` ... ``survey5``, ``hm1``, ``hm2``
    - Explicit range: ``91-99`` (optionally with ``OD`` prefixes)
    """
    raw = value.strip()
    normalized = re.sub(r"[\s_\-]+", " ", raw.lower()).strip()
    if normalized in MISSION_LENGTH_RANGES:
        return MISSION_LENGTH_RANGES[normalized]

    m = re.fullmatch(r"(?:od)?\s*(\d+)\s*-\s*(?:od)?\s*(\d+)", raw, flags=re.IGNORECASE)
    if m is None:
        known = ", ".join(sorted(MISSION_LENGTH_RANGES.keys()))
        raise ValueError(
            f"Unsupported mission_length={value!r}. Use one of [{known}] or an explicit range like '91-99'."
        )

    od_start = int(m.group(1))
    od_end = int(m.group(2))
    if od_start > od_end:
        raise ValueError(f"Invalid mission_length={value!r}: start OD must be <= end OD")
    return od_start, od_end


def extract_od_from_pointing_filename(path: Path) -> int:
    """Extract OD number from a pointing filename stem.

    Uses the last contiguous digit block in the stem, e.g.:
    - ``processed_od_0091`` -> 91
    - ``pointing-0092`` -> 92
    """
    matches = re.findall(r"(\d+)", path.stem)
    if not matches:
        raise ValueError(f"Cannot infer OD from filename: {path.name}")
    return int(matches[-1])


def filter_pointing_files_by_mission_length(files: list[Path], mission_length: str | None) -> list[Path]:
    """Filter discovered pointing files to the requested mission-length range."""
    if mission_length is None or mission_length.strip() == "":
        return files

    od_start, od_end = parse_mission_length(mission_length)
    out: list[Path] = []
    for p in files:
        od = extract_od_from_pointing_filename(p)
        if od_start <= od <= od_end:
            out.append(p)
    return out


def build_pointing_file_paths(pointings_prefix: str, od_start: int, od_end: int) -> list[Path]:
    """Build the list of pointing NPZ paths for an OD range.

    Constructs paths of the form ``{pointings_prefix}od_{od:04d}.npz`` for each
    OD in ``[od_start, od_end]`` (inclusive), returning only paths that exist.

    Args:
        pointings_prefix: Prefix for pointing files, e.g.
            ``"inputs/pointings/pointing_"``.
        od_start: First operational day (inclusive).
        od_end: Last operational day (inclusive).

    Returns:
        Sorted list of existing :class:`pathlib.Path` objects.

    Raises:
        FileNotFoundError: If no files exist for the requested range.
    """
    candidates = [Path(f"{pointings_prefix}od_{od:04d}.npz") for od in range(od_start, od_end + 1)]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        raise FileNotFoundError(
            f"No pointing files found for pointings={pointings_prefix!r}, OD{od_start}-OD{od_end}"
        )
    return existing


def _format_od_ranges(ods: list[int]) -> str:
    """Format a sorted list of OD numbers as compact ranges.

    Consecutive ODs are collapsed with ``-``; gaps are separated by ``,``.
    E.g. ``[91,92,93,95,98,99]`` → ``"91-93,95,98-99"``.
    """
    if not ods:
        return ""
    parts: list[str] = []
    start = prev = ods[0]
    for od in ods[1:]:
        if od == prev + 1:
            prev = od
        else:
            parts.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = od
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def print_mpi_distribution(
    comm,
    rank: int,
    size: int,
    local_ods: list[int] | None = None,
) -> None:
    """Print the MPI distribution at the start of a run.

    Gathers the hostname and assigned OD list from every rank and prints a
    summary on rank 0.  If the run is serial (``size == 1``) a single
    informational line is printed instead.

    Args:
        comm: The ``mpi4py`` communicator, or ``None`` for a serial run.
        rank: MPI rank of the calling process.
        size: Total number of MPI ranks.
        local_ods: OD numbers assigned to this rank (optional).
    """
    hostname = socket.gethostname()

    if size == 1:
        msg = f"[MPI] Serial run on host {hostname}"
        if local_ods:
            ods_sorted = sorted(local_ods)
            od_info = f"ODs {_format_od_ranges(ods_sorted)} ({len(ods_sorted)} ODs)"
            msg += f" | {od_info}"
        print(msg, flush=True)
        return

    hostnames: list[str] = comm.gather(hostname, root=0)
    all_ods: list[list[int] | None] = comm.gather(local_ods, root=0)
    if rank == 0:
        width = len(str(size - 1))
        print(f"[MPI] Parallel run: {size} ranks", flush=True)
        for r, h in enumerate(hostnames):
            ods = all_ods[r] if all_ods is not None else None
            if ods:
                ods_sorted = sorted(ods)
                od_info = f"ODs {_format_od_ranges(ods_sorted)} ({len(ods_sorted)} ODs)"
            else:
                od_info = "no ODs assigned"
            print(f"  rank {r:>{width}} : {h} | {od_info}", flush=True)


def estimate_memory_per_rank_mb(nside: int, lmax: int = 0, mmax: int = 0) -> float:
    """Estimate the peak memory per MPI rank in MB.

    Accounts for the full-sky normal-equation matrix, hit map, output maps, and —
    when *lmax* > 0 — one ``ducc0.totalconvolve.Interpolator`` instance (the dominant
    term at high *lmax* / *mmax*).

    The Interpolator estimate is a lower bound assuming complex64 (float32) internal
    storage and no NUFFT oversampling: ``(lmax+1) × 2(lmax+1) × (2*mmax+1) × 8 B``.
    The actual size can be 1.5–2× larger depending on the epsilon target.

    Args:
        nside: HEALPix resolution parameter.
        lmax: Maximum multipole (0 → Interpolator term omitted).
        mmax: Maximum beam azimuthal order (beam kmax).

    Returns:
        Estimated peak memory in MB.
    """
    npix = 12 * nside * nside
    matrix_mb = npix * 9 * 8 / 1024**2   # (npix, 3, 3) float64
    hits_mb   = npix * 8 / 1024**2        # (npix,) int64
    maps_mb   = npix * 3 * 8 / 1024**2   # t, q, u output maps
    interp_mb = 0.0
    if lmax > 0:
        # One ducc0 Interpolator: internal grid (lmax+1) × 2*(lmax+1) × (2*mmax+1)
        # stored as complex64 (8 B) — conservative lower bound, no oversampling.
        interp_mb = (lmax + 1) * 2 * (lmax + 1) * (2 * mmax + 1) * 8 / 1024**2
    return matrix_mb + hits_mb + maps_mb + interp_mb


def suggest_tasks_per_node(
    nside: int,
    node_memory_mb: float,
    cores_per_node: int,
    lmax: int = 0,
    mmax: int = 0,
) -> int:
    """Suggest the maximum number of MPI tasks per node for a given nside.

    Args:
        nside: HEALPix resolution parameter.
        node_memory_mb: Total physical memory on one compute node in MB.
        cores_per_node: Number of cores (and therefore maximum tasks) per node.
        lmax: Maximum multipole (passed to :func:`estimate_memory_per_rank_mb`).
        mmax: Maximum beam azimuthal order (passed to :func:`estimate_memory_per_rank_mb`).

    Returns:
        Recommended number of MPI tasks per node (capped at ``cores_per_node``).
    """
    mem_per_rank = estimate_memory_per_rank_mb(nside, lmax=lmax, mmax=mmax)
    max_tasks = int(node_memory_mb / mem_per_rank)
    return min(max_tasks, cores_per_node)


def resolve_nthreads(nthreads: int) -> int:
    """Resolve the effective number of threads to use for both ducc0 and numba.

    Convention:
    - ``nthreads == 0``: read ``OMP_NUM_THREADS`` from the environment;
      fall back to 1 if the variable is unset or invalid.
    - ``nthreads > 0``: use the given value as-is.

    Args:
        nthreads: Value from the ``convolution.nthreads`` config key.

    Returns:
        Resolved thread count (always >= 1).
    """
    if nthreads != 0:
        return max(1, int(nthreads))
    raw = os.environ.get("OMP_NUM_THREADS", "")
    try:
        return max(1, int(raw))
    except ValueError:
        return 1
