# -*- coding: utf-8 -*-
# Copyright 2017-2019 The diffsims developers
#
# This file is part of diffsims.
#
# diffsims is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# diffsims is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with diffsims.  If not, see <http://www.gnu.org/licenses/>.

"""
Scattering Paramaters as Tabulated in
"Advanced Computing in Electron Microscopy - Second Edition (2010) - Earl.J.Kirkland"
ISBN 978-1-4419-6532-5
Pages 253-260 Appendix C

This transcription comes from scikit-ued (MIT license) - https://pypi.python.org/pypi/scikit-ued
"""

import numpy as np

# We refer to Kirkland's work for interpretations, but the ordering here is:
# chisq, a1, b1, a2, b2, a3, b3, c1, d1, c2, d2, c3, d3

scattering_params = {
    1: np.array([0.170190, 4.20298324e-003, 2.25350888e-001, 6.27762505e-002, 2.25366950e-001, 3.00907347e-002, 2.25331756e-001, 6.77756695e-002, 4.38854001e+000, 3.56609237e-003, 4.03884823e-001, 2.76135815e-002, 1.44490166e+000]),
    2: np.array([0.396634, 1.87543704e-005, 2.12427997e-001, 4.10595800e-004, 3.32212279e-001, 1.96300059e-001, 5.17325152e-001, 8.36015738e-003, 3.66668239e-001, 2.95102022e-002, 1.37171827e+000, 4.65928982e-007, 3.75768025e+004]),
    3: np.array([0.286232, 7.45843816e-002, 8.81151424e-001, 7.15382250e-002, 4.59142904e-002, 1.45315229e-001, 8.81301714e-001, 1.12125769e+000, 1.88483665e+001, 2.51736525e-003, 1.59189995e-001, 3.58434971e-001, 6.12371000e+000]),
    4: np.array([0.195442, 6.11642897e-002, 9.90182132e-002, 1.25755034e-001, 9.90272412e-002, 2.00831548e-001, 1.87392509e+000, 7.87242876e-001, 9.32794929e+000, 1.58847850e-003, 8.91900236e-002, 2.73962031e-001, 3.20687658e+000]),
    5: np.array([0.146989, 1.25716066e-001, 1.48258830e-001, 1.73314452e-001, 1.48257216e-001, 1.84774811e-001, 3.34227311e+000, 1.95250221e-001, 1.97339463e+000, 5.29642075e-001, 5.70035553e+000, 1.08230500e-003, 5.64857237e-002]),
    6: np.array([0.102440, 2.12080767e-001, 2.08605417e-001, 1.99811865e-001, 2.08610186e-001, 1.68254385e-001, 5.57870773e+000, 1.42048360e-001, 1.33311887e+000, 3.63830672e-001, 3.80800263e+000, 8.35012044e-004, 4.03982620e-002]),
    7: np.array([0.060249, 5.33015554e-001, 2.90952515e-001, 5.29008883e-002, 1.03547896e+001, 9.24159648e-002, 1.03540028e+001, 2.61799101e-001, 2.76252723e+000, 8.80262108e-004, 3.47681236e-002, 1.10166555e-001, 9.93421736e-001]),
    8: np.array([0.039944, 3.39969204e-001, 3.81570280e-001, 3.07570172e-001, 3.81571436e-001, 1.30369072e-001, 1.91919745e+001, 8.83326058e-002, 7.60635525e-001, 1.96586700e-001, 2.07401094e+000, 9.96220028e-004, 3.03266869e-002]),
    9: np.array([0.027866, 2.30560593e-001, 4.80754213e-001, 5.26889648e-001, 4.80763895e-001, 1.24346755e-001, 3.95306720e+001, 1.24616894e-003, 2.62181803e-002, 7.20452555e-002, 5.92495593e-001, 1.53075777e-001, 1.59127671e+000]),
    10: np.array([0.021836, 4.08371771e-001, 5.88228627e-001, 4.54418858e-001, 5.88288655e-001, 1.44564923e-001, 1.21246013e+002, 5.91531395e-002, 4.63963540e-001, 1.24003718e-001, 1.23413025e+000, 1.64986037e-003, 2.05869217e-002]),
    11: np.array([0.064136, 1.36471662e-001, 4.99965301e-002, 7.70677865e-001, 8.81899664e-001, 1.56862014e-001, 1.61768579e+001, 9.96821513e-001, 2.00132610e+001, 3.80304670e-002, 2.60516254e-001, 1.27685089e-001, 6.99559329e-001]),
    12: np.array([0.051303, 3.04384121e-001, 8.42014377e-002, 7.56270563e-001, 1.64065598e+000, 1.01164809e-001, 2.97142975e+001, 3.45203403e-002, 2.16596094e-001, 9.71751327e-001, 1.21236852e+001, 1.20593012e-001, 5.60865838e-001]),
    13: np.array([0.049529, 7.77419424e-001, 2.71058227e+000, 5.78312036e-002, 7.17532098e+001, 4.26386499e-001, 9.13331555e-002, 1.13407220e-001, 4.48867451e-001, 7.90114035e-001, 8.66366718e+000, 3.23293496e-002, 1.78503463e-001]),
    14: np.array([0.071667, 1.06543892e+000, 1.04118455e+000, 1.20143691e-001, 6.87113368e+001, 1.80915263e-001, 8.87533926e-002, 1.12065620e+000, 3.70062619e+000, 3.05452816e-002, 2.14097897e-001, 1.59963502e+000, 9.99096638e+000]),
    15: np.array([0.047673, 1.05284447e+000, 1.31962590e+000, 2.99440284e-001, 1.28460520e-001, 1.17460748e-001, 1.02190163e+002, 9.60643452e-001, 2.87477555e+000, 2.63555748e-002, 1.82076844e-001, 1.38059330e+000, 7.49165526e+000]),
    16: np.array([0.033482, 1.01646916e+000, 1.69181965e+000, 4.41766748e-001, 1.74180288e-001, 1.21503863e-001, 1.67011091e+002, 8.27966670e-001, 2.30342810e+000, 2.33022533e-002, 1.56954150e-001, 1.18302846e+000, 5.85782891e+000]),
    17: np.array([0.206186, 9.44221116e-001, 2.40052374e-001, 4.37322049e-001, 9.30510439e+000, 2.54547926e-001, 9.30486346e+000, 5.47763323e-002, 1.68655688e-001, 8.00087488e-001, 2.97849774e+000, 1.07488641e-002, 6.84240646e-002]),
    18: np.array([0.263904, 1.06983288e+000, 2.87791022e-001, 4.24631786e-001, 1.24156957e+001, 2.43897949e-001, 1.24158868e+001, 4.79446296e-002, 1.36979796e-001, 7.64958952e-001, 2.43940729e+000, 8.23128431e-003, 5.27258749e-002]),
    19: np.array([0.161900, 6.92717865e-001, 7.10849990e+000, 9.65161085e-001, 3.57532901e-001, 1.48466588e-001, 3.93763275e-002, 2.64645027e-002, 1.03591321e-001, 1.80883768e+000, 3.22845199e+001, 5.43900018e-001, 1.67791374e+000]),
    20: np.array([0.085209, 3.66902871e-001, 6.14274129e-002, 8.66378999e-001, 5.70881727e-001, 6.67203300e-001, 7.82965639e+000, 4.87743636e-001, 1.32531318e+000, 1.82406314e+000, 2.10056032e+001, 2.20248453e-002, 9.11853450e-002]),
    21: np.array([0.052352, 3.78871777e-001, 6.98910162e-002, 9.00022505e-001, 5.21061541e-001, 7.15288914e-001, 7.87707920e+000, 1.88640973e-002, 8.17512708e-002, 4.07945949e-001, 1.11141388e+000, 1.61786540e+000, 1.80840759e+001]),
    22: np.array([0.035298, 3.62383267e-001, 7.54707114e-002, 9.84232966e-001, 4.97757309e-001, 7.41715642e-001, 8.17659391e+000, 3.62555269e-001, 9.55524906e-001, 1.49159390e+000, 1.62221677e+001, 1.61659509e-002, 7.33140839e-002]),
    23: np.array([0.030745, 3.52961378e-001, 8.19204103e-002, 7.46791014e-001, 8.81189511e+000, 1.08364068e+000, 5.10646075e-001, 1.39013610e+000, 1.48901841e+001, 3.31273356e-001, 8.38543079e-001, 1.40422612e-002, 6.57432678e-002]),
    24: np.array([0.015287, 1.34348379e+000, 1.25814353e+000, 5.07040328e-001, 1.15042811e+001, 4.26358955e-001, 8.53660389e-002, 1.17241826e-002, 6.00177061e-002, 5.11966516e-001, 1.53772451e+000, 3.38285828e-001, 6.62418319e-001]),
    25: np.array([0.031274, 3.26697613e-001, 8.88813083e-002, 7.17297000e-001, 1.11300198e+001, 1.33212464e+000, 5.82141104e-001, 2.80801702e-001, 6.71583145e-001, 1.15499241e+000, 1.26825395e+001, 1.11984488e-002, 5.32334467e-002]),
    26: np.array([0.031315, 3.13454847e-001, 8.99325756e-002, 6.89290016e-001, 1.30366038e+001, 1.47141531e+000, 6.33345291e-001, 1.03298688e+000, 1.16783425e+001, 2.58280285e-001, 6.09116446e-001, 1.03460690e-002, 4.81610627e-002]),
    27: np.array([0.031643, 3.15878278e-001, 9.46683246e-002, 1.60139005e+000, 6.99436449e-001, 6.56394338e-001, 1.56954403e+001, 9.36746624e-001, 1.09392410e+001, 9.77562646e-003, 4.37446816e-002, 2.38378578e-001, 5.56286483e-001]),
    28: np.array([0.032245, 1.72254630e+000, 7.76606908e-001, 3.29543044e-001, 1.02262360e-001, 6.23007200e-001, 1.94156207e+001, 9.43496513e-003, 3.98684596e-002, 8.54063515e-001, 1.04078166e+001, 2.21073515e-001, 5.10869330e-001]),
    29: np.array([0.010467, 3.58774531e-001, 1.06153463e-001, 1.76181348e+000, 1.01640995e+000, 6.36905053e-001, 1.53659093e+001, 7.44930667e-003, 3.85345989e-002, 1.89002347e-001, 3.98427790e-001, 2.29619589e-001, 9.01419843e-001]),
    30: np.array([0.026698, 5.70893973e-001, 1.26534614e-001, 1.98908856e+000, 2.17781965e+000, 3.06060585e-001, 3.78619003e+001, 2.35600223e-001, 3.67019041e-001, 3.97061102e-001, 8.66419596e-001, 6.85657228e-003, 3.35778823e-002]),
    31: np.array([0.008110, 6.25528464e-001, 1.10005650e-001, 2.05302901e+000, 2.41095786e+000, 2.89608120e-001, 4.78685736e+001, 2.07910594e-001, 3.27807224e-001, 3.45079617e-001, 7.43139061e-001, 6.55634298e-003, 3.09411369e-002]),
    32: np.array([0.032198, 5.90952690e-001, 1.18375976e-001, 5.39980660e-001, 7.18937433e+001, 2.00626188e+000, 1.39304889e+000, 7.49705041e-001, 6.89943350e+000, 1.83581347e-001, 3.64667232e-001, 9.52190743e-003, 2.69888650e-002]),
    33: np.array([0.034014, 7.77875218e-001, 1.50733157e-001, 5.93848150e-001, 1.42882209e+002, 1.95918751e+000, 1.74750339e+000, 1.79880226e-001, 3.31800852e-001, 8.63267222e-001, 5.85490274e+000, 9.59053427e-003, 2.33777569e-002]),
    34: np.array([0.035703, 9.58390681e-001, 1.83775557e-001, 6.03851342e-001, 1.96819224e+002, 1.90828931e+000, 2.15082053e+000, 1.73885956e-001, 3.00006024e-001, 9.35265145e-001, 4.92471215e+000, 8.62254658e-003, 2.12308108e-002]),
    35: np.array([0.039250, 1.14136170e+000, 2.18708710e-001, 5.18118737e-001, 1.93916682e+002, 1.85731975e+000, 2.65755396e+000, 1.68217399e-001, 2.71719918e-001, 9.75705606e-001, 4.19482500e+000, 7.24187871e-003, 1.99325718e-002]),
    36: np.array([0.045421, 3.24386970e-001, 6.31317973e+001, 1.31732163e+000, 2.54706036e-001, 1.79912614e+000, 3.23668394e+000, 4.29961425e-003, 1.98965610e-002, 1.00429433e+000, 3.61094513e+000, 1.62188197e-001, 2.45583672e-001]),
    37: np.array([0.130044, 2.90445351e-001, 3.68420227e-002, 2.44201329e+000, 1.16013332e+000, 7.69435449e-001, 1.69591472e+001, 1.58687000e+000, 2.53082574e+000, 2.81617593e-003, 1.88577417e-002, 1.28663830e-001, 2.10753969e-001]),
    38: np.array([0.188055, 1.37373086e-002, 1.87469061e-002, 1.97548672e+000, 6.36079230e+000, 1.59261029e+000, 2.21992482e-001, 1.73263882e-001, 2.01624958e-001, 4.66280378e+000, 2.53027803e+001, 1.61265063e-003, 1.53610568e-002]),
    39: np.array([0.174927, 6.75302747e-001, 6.54331847e-002, 4.70286720e-001, 1.06108709e+002, 2.63497677e+000, 2.06643540e+000, 1.09621746e-001, 1.93131925e-001, 9.60348773e-001, 1.63310938e+000, 5.28921555e-003, 1.66083821e-002]),
    40: np.array([0.072078, 2.64365505e+000, 2.20202699e+000, 5.54225147e-001, 1.78260107e+002, 7.61376625e-001, 7.67218745e-002, 6.02946891e-003, 1.55143296e-002, 9.91630530e-002, 1.76175995e-001, 9.56782020e-001, 1.54330682e+000]),
    41: np.array([0.011800, 6.59532875e-001, 8.66145490e-002, 1.84545854e+000, 5.94774398e+000, 1.25584405e+000, 6.40851475e-001, 1.22253422e-001, 1.66646050e-001, 7.06638328e-001, 1.62853268e+000, 2.62381591e-003, 8.26257859e-003]),
    42: np.array([0.008976, 6.10160120e-001, 9.11628054e-002, 1.26544000e+000, 5.06776025e-001, 1.97428762e+000, 5.89590381e+000, 6.48028962e-001, 1.46634108e+000, 2.60380817e-003, 7.84336311e-003, 1.13887493e-001, 1.55114340e-001]),
    43: np.array([0.023771, 8.55189183e-001, 1.02962151e-001, 1.66219641e+000, 7.64907000e+000, 1.45575475e+000, 1.01639987e+000, 1.05445664e-001, 1.42303338e-001, 7.71657112e-001, 1.34659349e+000, 2.20992635e-003, 7.90358976e-003]),
    44: np.array([0.010613, 4.70847093e-001, 9.33029874e-002, 1.58180781e+000, 4.52831347e-001, 2.02419818e+000, 7.11489023e+000, 1.97036257e-003, 7.56181595e-003, 6.26912639e-001, 1.25399858e+000, 1.02641320e-001, 1.33786087e-001]),
    45: np.array([0.012895, 4.20051553e-001, 9.38882628e-002, 1.76266507e+000, 4.64441687e-001, 2.02735641e+000, 8.19346046e+000, 1.45487176e-003, 7.82704517e-003, 6.22809600e-001, 1.17194153e+000, 9.91529915e-002, 1.24532839e-001]),
    46: np.array([0.009172, 2.10475155e+000, 8.68606470e+000, 2.03884487e+000, 3.78924449e-001, 1.82067264e-001, 1.42921634e-001, 9.52040948e-002, 1.17125900e-001, 5.91445248e-001, 1.07843808e+000, 1.13328676e-003, 7.80252092e-003]),
    47: np.array([0.006648, 2.07981390e+000, 9.92540297e+000, 4.43170726e-001, 1.04920104e-001, 1.96515215e+000, 6.40103839e-001, 5.96130591e-001, 8.89594790e-001, 4.78016333e-001, 1.98509407e+000, 9.46458470e-002, 1.12744464e-001]),
    48: np.array([0.005588, 1.63657549e+000, 1.24540381e+001, 2.17927989e+000, 1.45134660e+000, 7.71300690e-001, 1.26695757e-001, 6.64193880e-001, 7.77659202e-001, 7.64563285e-001, 1.66075210e+000, 8.61126689e-002, 1.05728357e-001]),
    49: np.array([0.002569, 2.24820632e+000, 1.51913507e+000, 1.64706864e+000, 1.30113424e+001, 7.88679265e-001, 1.06128184e-001, 8.12579069e-002, 9.94045620e-002, 6.68280346e-001, 1.49742063e+000, 6.38467475e-001, 7.18422635e-001]),
    50: np.array([0.005051, 2.16644620e+000, 1.13174909e+001, 6.88691021e-001, 1.10131285e-001, 1.92431751e+000, 6.74464853e-001, 5.65359888e-001, 7.33564610e-001, 9.18683861e-001, 1.02310312e+001, 7.80542213e-002, 9.31104308e-002]),
    51: np.array([0.004383, 1.73662114e+000, 8.84334719e-001, 9.99871380e-001, 1.38462121e-001, 2.13972409e+000, 1.19666432e+001, 5.60566526e-001, 6.72672880e-001, 9.93772747e-001, 8.72330411e+000, 7.37374982e-002, 8.78577715e-002]),
    52: np.array([0.004105, 2.09383882e+000, 1.26856869e+001, 1.56940519e+000, 1.21236537e+000, 1.30941993e+000, 1.66633292e-001, 6.98067804e-002, 8.30817576e-002, 1.04969537e+000, 7.43147857e+000, 5.55594354e-001, 6.17487676e-001]),
    53: np.array([0.004068, 1.60186925e+000, 1.95031538e-001, 1.98510264e+000, 1.36976183e+001, 1.48226200e+000, 1.80304795e+000, 5.53807199e-001, 5.67912340e-001, 1.11728722e+000, 6.40879878e+000, 6.60720847e-002, 7.86615429e-002]),
    54: np.array([0.004381, 1.60015487e+000, 2.92913354e+000, 1.71644581e+000, 1.55882990e+001, 1.84968351e+000, 2.22525983e-001, 6.23813648e-002, 7.45581223e-002, 1.21387555e+000, 5.56013271e+000, 5.54051946e-001, 5.21994521e-001]),
    55: np.array([0.042676, 2.95236854e+000, 6.01461952e+000, 4.28105721e-001, 4.64151246e+001, 1.89599233e+000, 1.80109756e-001, 5.48012938e-002, 7.12799633e-002, 4.70838600e+000, 4.56702799e+001, 5.90356719e-001, 4.70236310e-001]),
    56: np.array([0.043267, 3.19434243e+000, 9.27352241e+000, 1.98289586e+000, 2.28741632e-001, 1.55121052e-001, 3.82000231e-002, 6.73222354e-002, 7.30961745e-002, 4.48474211e+000, 2.95703565e+001, 5.42674414e-001, 4.08647015e-001]),
    57: np.array([0.033249, 2.05036425e+000, 2.20348417e-001, 1.42114311e-001, 3.96438056e-002, 3.23538151e+000, 9.56979169e+000, 6.34683429e-002, 6.92443091e-002, 3.97960586e+000, 2.53178406e+001, 5.20116711e-001, 3.83614098e-001]),
    58: np.array([0.029355, 3.22990759e+000, 9.94660135e+000, 1.57618307e-001, 4.15378676e-002, 2.13477838e+000, 2.40480572e-001, 5.01907609e-001, 3.66252019e-001, 3.80889010e+000, 2.43275968e+001, 5.96625028e-002, 6.59653503e-002]),
    59: np.array([0.029725, 1.58189324e-001, 3.91309056e-002, 3.18141995e+000, 1.04139545e+001, 2.27622140e+000, 2.81671757e-001, 3.97705472e+000, 2.61872978e+001, 5.58448277e-002, 6.30921695e-002, 4.85207954e-001, 3.54234369e-001]),
    60: np.array([0.027597, 1.81379417e-001, 4.37324793e-002, 3.17616396e+000, 1.07842572e+001, 2.35221519e+000, 3.05571833e-001, 3.83125763e+000, 2.54745408e+001, 5.25889976e-002, 6.02676073e-002, 4.70090742e-001, 3.39017003e-001]),
    61: np.array([0.025208, 1.92986811e-001, 4.37785970e-002, 2.43756023e+000, 3.29336996e-001, 3.17248504e+000, 1.11259996e+001, 3.58105414e+000, 2.46709586e+001, 4.56529394e-001, 3.24990282e-001, 4.94812177e-002, 5.76553100e-002]),
    62: np.array([0.023540, 2.12002595e-001, 4.57703608e-002, 3.16891754e+000, 1.14536599e+001, 2.51503494e+000, 3.55561054e-001, 4.44080845e-001, 3.11953363e-001, 3.36742101e+000, 2.40291435e+001, 4.65652543e-002, 5.52266819e-002]),
    63: np.array([0.022204, 2.59355002e+000, 3.82452612e-001, 3.16557522e+000, 1.17675155e+001, 2.29402652e-001, 4.76642249e-002, 4.32257780e-001, 2.99719833e-001, 3.17261920e+000, 2.34462738e+001, 4.37958317e-002, 5.29440680e-002]),
    64: np.array([0.017492, 3.19144939e+000, 1.20224655e+001, 2.55766431e+000, 4.08338876e-001, 3.32681934e-001, 5.85819814e-002, 4.14243130e-002, 5.06771477e-002, 2.61036728e+000, 1.99344244e+001, 4.20526863e-001, 2.85686240e-001]),
    65: np.array([0.020036, 2.59407462e-001, 5.04689354e-002, 3.16177855e+000, 1.23140183e+001, 2.75095751e+000, 4.38337626e-001, 2.79247686e+000, 2.23797309e+001, 3.85931001e-002, 4.87920992e-002, 4.10881708e-001, 2.77622892e-001]),
    66: np.array([0.019351, 3.16055396e+000, 1.25470414e+001, 2.82751709e+000, 4.67899094e-001, 2.75140255e-001, 5.23226982e-002, 4.00967160e-001, 2.67614884e-001, 2.63110834e+000, 2.19498166e+001, 3.61333817e-002, 4.68871497e-002]),
    67: np.array([0.018720, 2.88642467e-001, 5.40507687e-002, 2.90567296e+000, 4.97581077e-001, 3.15960159e+000, 1.27599505e+001, 3.91280259e-001, 2.58151831e-001, 2.48596038e+000, 2.15400972e+001, 3.37664478e-002, 4.50664323e-002]),
    68: np.array([0.018677, 3.15573213e+000, 1.29729009e+001, 3.11519560e-001, 5.81399387e-002, 2.97722406e+000, 5.31213394e-001, 3.81563854e-001, 2.49195776e-001, 2.40247532e+000, 2.13627616e+001, 3.15224214e-002, 4.33253257e-002]),
    69: np.array([0.018176, 3.15591970e+000, 1.31232407e+001, 3.22544710e-001, 5.97223323e-002, 3.05569053e+000, 5.61876773e-001, 2.92845100e-002, 4.16534255e-002, 3.72487205e-001, 2.40821967e-001, 2.27833695e+000, 2.10034185e+001]),
    70: np.array([0.018460, 3.10794704e+000, 6.06347847e-001, 3.14091221e+000, 1.33705269e+001, 3.75660454e-001, 7.29814740e-002, 3.61901097e-001, 2.32652051e-001, 2.45409082e+000, 2.12695209e+001, 2.72383990e-002, 3.99969597e-002]),
    71: np.array([0.015021, 3.11446863e+000, 1.38968881e+001, 5.39634353e-001, 8.91708508e-002, 3.06460915e+000, 6.79919563e-001, 2.58563745e-002, 3.82808522e-002, 2.13983556e+000, 1.80078788e+001, 3.47788231e-001, 2.22706591e-001]),
    72: np.array([0.012070, 3.01166899e+000, 7.10401889e-001, 3.16284788e+000, 1.38262192e+001, 6.33421771e-001, 9.48486572e-002, 3.41417198e-001, 2.14129678e-001, 1.53566013e+000, 1.55298698e+001, 2.40723773e-002, 3.67833690e-002]),
    73: np.array([0.010775, 3.20236821e+000, 1.38446369e+001, 8.30098413e-001, 1.18381581e-001, 2.86552297e+000, 7.66369118e-001, 2.24813887e-002, 3.52934622e-002, 1.40165263e+000, 1.46148877e+001, 3.33740596e-001, 2.05704486e-001]),
    74: np.array([0.009479, 9.24906855e-001, 1.28663377e-001, 2.75554557e+000, 7.65826479e-001, 3.30440060e+000, 1.34471170e+001, 3.29973862e-001, 1.98218895e-001, 1.09916444e+000, 1.35087534e+001, 2.06498883e-002, 3.38918459e-002]),
    75: np.array([0.004620, 1.96952105e+000, 4.98830620e+001, 1.21726619e+000, 1.33243809e-001, 4.10391685e+000, 1.84396916e+000, 2.90791978e-002, 2.84192813e-002, 2.30696669e-001, 1.90968784e-001, 6.08840299e-001, 1.37090356e+000]),
    76: np.array([0.003085, 2.06385867e+000, 4.05671697e+001, 1.29603406e+000, 1.46559047e-001, 3.96920673e+000, 1.82561596e+000, 2.69835487e-002, 2.84172045e-002, 2.31083999e-001, 1.79765184e-001, 6.30466774e-001, 1.38911543e+000]),
    77: np.array([0.003924, 2.21522726e+000, 3.24464090e+001, 1.37573155e+000, 1.60920048e-001, 3.78244405e+000, 1.78756553e+000, 2.44643240e-002, 2.82909938e-002, 2.36932016e-001, 1.70692368e-001, 6.48471412e-001, 1.37928390e+000]),
    78: np.array([0.003817, 9.84697940e-001, 1.60910839e-001, 2.73987079e+000, 7.18971667e-001, 3.61696715e+000, 1.29281016e+001, 3.02885602e-001, 1.70134854e-001, 2.78370726e-001, 1.49862703e+000, 1.52124129e-002, 2.83510822e-002]),
    79: np.array([0.003143, 9.61263398e-001, 1.70932277e-001, 3.69581030e+000, 1.29335319e+001, 2.77567491e+000, 6.89997070e-001, 2.95414176e-001, 1.63525510e-001, 3.11475743e-001, 1.39200901e+000, 1.43237267e-002, 2.71265337e-002]),
    80: np.array([0.002717, 1.29200491e+000, 1.83432865e-001, 2.75161478e+000, 9.42368371e-001, 3.49387949e+000, 1.46235654e+001, 2.77304636e-001, 1.55110144e-001, 4.30232810e-001, 1.28871670e+000, 1.48294351e-002, 2.61903834e-002]),
    81: np.array([0.003492, 3.75964730e+000, 1.35041513e+001, 3.21195904e+000, 6.66330993e-001, 6.47767825e-001, 9.22518234e-002, 2.76123274e-001, 1.50312897e-001, 3.18838810e-001, 1.12565588e+000, 1.31668419e-002, 2.48879842e-002]),
    82: np.array([0.001158, 1.00795975e+000, 1.17268427e-001, 3.09796153e+000, 8.80453235e-001, 3.61296864e+000, 1.47325812e+001, 2.62401476e-001, 1.43491014e-001, 4.05621995e-001, 1.04103506e+000, 1.31812509e-002, 2.39575415e-002]),
    83: np.array([0.026436, 1.59826875e+000, 1.56897471e-001, 4.38233925e+000, 2.47094692e+000, 2.06074719e+000, 5.72438972e+001, 1.94426023e-001, 1.32979109e-001, 8.22704978e-001, 9.56532528e-001, 2.33226953e-002, 2.23038435e-002]),
    84: np.array([0.008962, 1.71463223e+000, 9.79262841e+001, 2.14115960e+000, 2.10193717e-001, 4.37512413e+000, 3.66948812e+000, 2.16216680e-002, 1.98456144e-002, 1.97843837e-001, 1.33758807e-001, 6.52047920e-001, 7.80432104e-001]),
    85: np.array([0.033776, 1.48047794e+000, 1.25943919e+002, 2.09174630e+000, 1.83803008e-001, 4.75246033e+000, 4.19890596e+000, 1.85643958e-002, 1.81383503e-002, 2.05859375e-001, 1.33035404e-001, 7.13540948e-001, 7.03031938e-001]),
    86: np.array([0.050132, 6.30022295e-001, 1.40909762e-001, 3.80962881e+000, 3.08515540e+001, 3.89756067e+000, 6.51559763e-001, 2.40755100e-001, 1.08899672e-001, 2.62868577e+000, 6.42383261e+000, 3.14285931e-002, 2.42346699e-002]),
    87: np.array([0.056720, 5.23288135e+000, 8.60599536e+000, 2.48604205e+000, 3.04543982e-001, 3.23431354e-001, 3.87759096e-002, 2.55403596e-001, 1.28717724e-001, 5.53607228e-001, 5.36977452e-001, 5.75278889e-003, 1.29417790e-002]),
    88: np.array([0.081498, 1.44192685e+000, 1.18740873e-001, 3.55291725e+000, 1.01739750e+000, 3.91259586e+000, 6.31814783e+001, 2.16173519e-001, 9.55806441e-002, 3.94191605e+000, 3.50602732e+001, 4.60422605e-002, 2.20850385e-002]),
    89: np.array([0.077643, 1.45864127e+000, 1.07760494e-001, 4.18945405e+000, 8.89090649e+001, 3.65866182e+000, 1.05088931e+000, 2.08479229e-001, 9.09335557e-002, 3.16528117e+000, 3.13297788e+001, 5.23892556e-002, 2.08807697e-002]),
    90: np.array([0.048096, 1.19014064e+000, 7.73468729e-002, 2.55380607e+000, 6.59693681e-001, 4.68110181e+000, 1.28013896e+001, 2.26121303e-001, 1.08632194e-001, 3.58250545e-001, 4.56765664e-001, 7.82263950e-003, 1.62623474e-002]),
    91: np.array([0.070186, 4.68537504e+000, 1.44503632e+001, 2.98413708e+000, 5.56438592e-001, 8.91988061e-001, 6.69512914e-002, 2.24825384e-001, 1.03235396e-001, 3.04444846e-001, 4.27255647e-001, 9.48162708e-003, 1.77730611e-002]),
    92: np.array([0.072478, 4.63343606e+000, 1.63377267e+001, 3.18157056e+000, 5.69517868e-001, 8.76455075e-001, 6.88860012e-002, 2.21685477e-001, 9.84254550e-002, 2.72917100e-001, 4.09470917e-001, 1.11737298e-002, 1.86215410e-002]),
    93: np.array([0.074792, 4.56773888e+000, 1.90992795e+001, 3.40325179e+000, 5.90099634e-001, 8.61841923e-001, 7.03204851e-002, 2.19728870e-001, 9.36334280e-002, 2.38176903e-001, 3.93554882e-001, 1.38306499e-002, 1.94437286e-002]),
    94: np.array([0.071877, 5.45671123e+000, 1.01892720e+001, 1.11687906e-001, 3.98131313e-002, 3.30260343e+000, 3.14622212e-001, 1.84568319e-001, 1.04220860e-001, 4.93644263e-001, 4.63080540e-001, 3.57484743e+000, 2.19369542e+001]),
    95: np.array([0.062156, 5.38321999e+000, 1.07289857e+001, 1.23343236e-001, 4.15137806e-002, 3.46469090e+000, 3.39326208e-001, 1.75437132e-001, 9.98932346e-002, 3.39800073e+000, 2.11601535e+001, 4.69459519e-001, 4.51996970e-001]),
    96: np.array([0.050111, 5.38402377e+000, 1.11211419e+001, 3.49861264e+000, 3.56750210e-001, 1.88039547e-001, 5.39853583e-002, 1.69143137e-001, 9.60082633e-002, 3.19595016e+000, 1.80694389e+001, 4.64393059e-001, 4.36318197e-001]),
    97: np.array([0.044081, 3.66090688e+000, 3.84420906e-001, 2.03054678e-001, 5.48547131e-002, 5.30697515e+000, 1.17150262e+001, 1.60934046e-001, 9.21020329e-002, 3.04808401e+000, 1.73525367e+001, 4.43610295e-001, 4.27132359e-001]),
    98: np.array([0.041053, 3.94150390e+000, 4.18246722e-001, 5.16915345e+000, 1.25201788e+001, 1.61941074e-001, 4.81540117e-002, 4.15299561e-001, 4.24913856e-001, 2.91761325e+000, 1.90899693e+001, 1.51474927e-001, 8.81568925e-002]),
    99: np.array([0.036478, 4.09780623e+000, 4.46021145e-001, 5.10079393e+000, 1.31768613e+001, 1.74617289e-001, 5.02742829e-002, 2.76774658e+000, 1.84815393e+001, 1.44496639e-001, 8.46232592e-002, 4.02772109e-001, 4.17640100e-001]),
    100: np.array([0.032651, 4.24934820e+000, 4.75263933e-001, 5.03556594e+000, 1.38570834e+001, 1.88920613e-001, 5.26975158e-002, 3.94356058e-001, 4.11193751e-001, 2.61213100e+000, 1.78537905e+001, 1.38001927e-001, 8.12774434e-002]),
    101: np.array([0.029668, 2.00942931e-001, 5.48366518e-002, 4.40119869e+000, 5.04248434e-001, 4.97250102e+000, 1.45721366e+001, 2.47530599e+000, 1.72978308e+001, 3.86883197e-001, 4.05043898e-001, 1.31936095e-001, 7.80821071e-002]),
    102: np.array([0.027320, 2.16052899e-001, 5.83584058e-002, 4.91106799e+000, 1.53264212e+001, 4.54862870e+000, 5.34434760e-001, 2.36114249e+000, 1.68164803e+001, 1.26277292e-001, 7.50304633e-002, 3.81364501e-001, 3.99305852e-001]),
    103: np.array([0.024894, 4.86738014e+000, 1.60320520e+001, 3.19974401e-001, 6.70871138e-002, 4.58872425e+000, 5.77039373e-001, 1.21482448e-001, 7.22275899e-002, 2.31639872e+000, 1.41279737e+001, 3.79258137e-001, 3.89973484e-001])}
