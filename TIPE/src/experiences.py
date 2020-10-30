import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
###Influence et temps en fonction de k pour DegreeDiscount
#DegreeDiscount sur HepPh, p = 0.01, iterations = 20000, k de 1 à 46
k1 = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46]
influences1 = [420, 442, 455, 462, 467, 472, 476, 481, 484, 488, 492, 494, 498, 502, 505, 509, 512, 514, 518, 521, 523, 526, 529, 532]
temps1 = [0.03741288185119629, 0.03955483436584473, 0.04516291618347168, 0.04876589775085449, 0.07937216758728027, 0.055667877197265625, 0.05630993843078613, 0.0577092170715332, 0.0593869686126709, 0.05682492256164551, 0.05741620063781738, 0.058921098709106445, 0.060561180114746094, 0.06161904335021973, 0.06213998794555664, 0.06495189666748047, 0.06368684768676758, 0.06567192077636719, 0.06583786010742188, 0.06625795364379883, 0.06799197196960449, 0.06863093376159668, 0.07221603393554688, 0.07291293144226074]

#DegreeDiscount sur HepPh, p = 0.01, iterations = 10000, k de 100 à 2100
k2 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100]
influences2 = [606, 727, 841, 951, 1060, 1168, 1274, 1380, 1484, 1587, 1691, 1793, 1897, 2000, 2102, 2223, 2335, 2442, 2544, 2648, 2752]

plt.plot(k1+k2, influences1+influences2)
plt.show()
##Influence en fonction de iterations
#Evaluation de l'influence sur HepPh, p = 0.01, k = 100, pour differents iterations
iterations = [50, 100, 200, 300, 400, 500] + [1000*k for k in range(1, 31)]
influences = [475, 477, 471, 472, 473, 475, 473, 475, 474, 475, 474, 474, 473, 473, 474, 474, 474, 474, 474, 474, 474, 474, 474, 473, 474, 474, 474, 474, 474, 474, 474, 474, 474, 474, 474, 474]
plt.plot(iterations , influences)
plt.show()

##
#CELF sur NetHEPT, p = 0.01, k de 1 à 46, :
influences = [4, 6, 8, 10, 12, 14, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73]
temps = 6882.288732767105
##
#CELF sur Simmons, p = 0.01, k de 1 à 200, IC
influences_CELF_IC = [9, 16, 23, 29, 35, 41, 47, 53, 58, 62, 66, 70, 74, 78, 81, 84, 87, 90, 93, 96, 99, 102, 104, 106, 108, 110, 113, 115, 117, 119, 121, 123, 125, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 155, 156, 157, 161, 163, 165, 167, 168, 169, 170, 171, 174, 175, 178, 179, 182, 183, 184, 185, 186, 188, 190, 191, 193, 195, 196, 197, 198, 199, 200, 202, 203, 205, 206, 207, 209, 210, 211, 212, 214, 216, 217, 218, 219,220, 221, 223, 224, 225, 226, 228, 230, 231, 233, 234, 235, 236, 237, 239, 240, 242, 243, 245, 246, 248, 249, 250, 251, 252, 253, 255, 256, 258, 259, 260, 261, 264, 265, 268, 269, 270, 271, 272, 275, 276, 277, 278, 280, 281, 282, 283, 285, 288, 289, 291, 292, 293, 294, 296, 297, 299, 300, 301, 302, 303,304,307,308, 309, 310, 311, 313, 314, 315, 316, 317, 320, 321, 322, 323, 324, 326, 327, 329, 330, 331, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 346, 347, 348, 349, 350,351,353,354,355,356,357,358, 359, 361, 362]

temps_CELF_IC = 1232.234116077423

etapes = [0, 6, 1, 3, 1, 1, 7, 91, 10, 0, 0, 20, 2, 63, 23, 44, 6, 35, 0, 3, 17, 67, 2, 2, 0, 4, 1, 5, 2, 3, 2, 3, 1, 118, 16, 7, 17, 22, 6, 1, 4, 3, 7, 36, 64, 265, 8, 4, 0, 0, 8, 2, 2, 1, 0, 0, 0, 0, 0, 0, 1, 0,4, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 3, 0, 1, 3, 3, 0, 0, 1, 1, 1, 2, 1, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 1,1, 1, 0, 0, 0, 0, 7, 2, 21, 1, 0, 1, 3, 2, 3, 11, 0, 0, 0, 0, 3, 4, 0, 2, 0, 0, 1, 2, 0, 0, 10, 2, 1, 0, 0, 3, 0, 2, 1, 0, 0, 3, 0, 4, 6, 0, 0, 0, 0, 2, 0, 0, 4, 0, 2, 0, 0, 0, 2, 2, 0, 0, 0,0, 0, 1]

#CELF sur Simmons, poids = 1/degre, k de 1 à 50, LT
influences_CELF_LT = [12.76, 21.944, 30.688, 35.016, 38.256, 45.792, 51.488, 55.384, 67.264, 72.624, 74.536, 76.568, 83.32, 88.016, 93.44, 99.552, 112.344, 115.872, 117.544, 121.344, 127.608, 134.096, 135.632, 137.936, 145.424, 146.888, 148.32, 155.856, 157.912, 161.056, 166.544, 169.448, 172.2, 175.648, 181.832, 184.096, 184.96, 186.776, 193.472, 194.608, 198.376, 201.744, 208.048, 209.648, 221.48, 223.072, 226.616, 234.368, 236.856, 244.248]
temps_CELF_LT = 4927.6006898880005

#CELF sur Simmons, p = 0.015, k de 1 à 50, IC
influences_CELF_IC_2 =[61, 90, 110, 133, 141, 149, 157, 165, 175, 183, 189, 193, 197, 201, 205, 209, 212, 215, 220, 223, 226, 229, 233, 236, 242, 245, 248, 251, 254, 257, 260, 262, 264, 266, 268, 270, 273, 
275, 277, 279, 281, 282, 283, 284, 287, 288, 290, 291, 292, 294]
k = [i for i in range(1, 51)]
plt.plot(k, influences_CELF_LT, 'red', label  = "Linear Threshold")
plt.plot(k, influences_CELF_IC[:50], 'blue', label = "Independent Cascade (p=0,01)")
plt.legend()
plt.show()

##
#Greedy sur Simmons, p = 0.01, k de 1 à 50, IC
temps_greedy = 29507

influences_greedy = [9, 17, 25, 30, 36, 41, 47, 52, 58, 62, 67, 70, 74, 78, 82, 85, 88, 92, 95, 98, 101, 104, 107, 109, 113, 116, 119, 121, 124, 127, 130, 132, 135, 136, 139, 141, 144, 146, 148, 150, 152, 154, 157, 158, 161, 163, 165, 168, 169, 171]
k = [i for i in range(1, 51)]

influences_CELF_IC = [9, 16, 23, 29, 35, 41, 47, 53, 58, 62, 66, 70, 74, 78, 81, 84, 87, 90, 93, 96, 99, 102, 104, 106, 108, 110, 113, 115, 117, 119, 121, 123, 125, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 155, 156, 157, 161, 163, 165, 167, 168, 169, 170, 171, 174, 175, 178, 179, 182, 183, 184, 185, 186, 188, 190, 191, 193, 195, 196, 197, 198, 199, 200, 202, 203, 205, 206, 207, 209, 210, 211, 212, 214, 216, 217, 218, 219,220, 221, 223, 224, 225, 226, 228, 230, 231, 233, 234, 235, 236, 237, 239, 240, 242, 243, 245, 246, 248, 249, 250, 251, 252, 253, 255, 256, 258, 259, 260, 261, 264, 265, 268, 269, 270, 271, 272, 275, 276, 277, 278, 280, 281, 282, 283, 285, 288, 289, 291, 292, 293, 294, 296, 297, 299, 300, 301, 302, 303,304,307,308, 309, 310, 311, 313, 314, 315, 316, 317, 320, 321, 322, 323, 324, 326, 327, 329, 330, 331, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 346, 347, 348, 349, 350,351,353,354,355,356,357,358, 359, 361, 362]

##


#DegreeDiscount sur Simmons, p = 0.01, k de 1 à 200, IC
influences_discount = [7.955, 16.966, 22.446, 28.093, 32.803, 36.642, 40.754, 43.917, 47.769, 50.668, 
55.224, 57.733, 61.269, 63.677, 67.565, 69.739, 70.085, 73.937, 76.573, 79.455, 
82.553, 83.591, 85.598, 87.601, 90.658, 92.851, 95.601, 96.594, 98.133, 100.714,
102.556, 103.918, 106.467, 106.854, 109.208, 111.728, 112.049, 114.483, 116.009
, 119.728, 120.387, 121.951, 122.438, 124.765, 127.585, 128.038, 129.125, 132.347, 134.604, 133.764, 135.995, 139.019, 138.015, 140.293, 142.143, 141.864, 145.042, 146.502, 148.127, 149.579, 151.705, 152.246, 152.666, 154.481, 157.728, 157.165, 159.493, 160.673, 162.458, 163.176, 164.761, 166.774, 166.915, 168.447, 169.516, 171.664, 172.01, 174.048, 174.717, 177.063, 178.339, 179.273, 180.851, 180.863, 183.867, 183.524, 185.142, 186.003, 188.503, 189.53, 188.75, 191.67, 192.346, 193.846, 195.848, 195.218, 196.921, 199.237, 200.052, 201.292, 202.425, 203.632, 205.251, 206.464, 207.747, 207.211, 210.675, 210.605, 211.747, 213.246, 214.133, 215.332, 216.519, 217.568, 219.667, 220.6, 221.365, 222.661, 222.779, 223.521, 224.481, 226.43, 227.787, 230.014, 229.822, 230.485, 231.93, 232.855, 235.143, 235.949, 236.585, 238.098, 238.987, 239.818, 241.812, 241.274, 243.241, 243.438, 245.508, 244.965, 248.003, 248.823, 250.317, 250.068, 252.068, 253.202, 253.521, 254.786, 256.063, 257.374, 258.069, 258.812, 259.895, 261.842, 263.034, 263.09, 264.48, 265.783, 265.827, 267.645, 268.953, 270.977, 270.936, 272.108, 271.702, 274.026, 276.784, 276.987, 279.007, 282.34, 284.041, 285.37, 287.31, 288.951, 290.674, 292.189, 294.508, 294.63, 297.775, 299.456, 302.698, 302.477, 304.233, 305.868, 309.12, 309.624, 311.15, 311.519, 314.104, 316.695, 318.05, 317.989
, 320.882, 319.494, 323.23, 322.622, 325.669, 326.028, 326.44, 328.9]

"""#SingleDiscount sur Simmons, p = 0.01, k de 1 à 200, IC
influences_single = [7.97, 18.063, 22.487, 27.919, 31.95, 36.351, 41.097, 44.413, 47.401, 51.614, 54.552, 57.278, 60.277, 63.693, 66.658, 69.553, 71.811, 74.313, 77.156, 79.042, 82
.21, 84.794, 85.553, 88.054, 90.887, 92.217, 95.003, 95.062, 98.301, 100.793, 10
2.323, 104.366, 105.537, 107.011, 108.964, 111.723, 114.452, 115.464, 116.532, 1
19.045, 119.027, 121.472, 124.157, 124.265, 127.971, 128.085, 128.909, 129.9, 13
2.577, 133.291, 135.914, 137.509, 138.708, 139.841, 142.297, 144.093, 145.761, 1
46.334, 147.729, 149.129, 150.291, 151.657, 153.646, 154.846, 158.115, 158.595, 
158.498, 161.104, 162.124, 163.877, 164.342, 165.944, 167.858, 168.828, 169.409,
 170.767, 172.259, 173.929, 174.258, 176.354, 177.616, 179.219, 179.789, 181.976
, 182.112, 185.237, 184.819, 185.687, 187.651, 189.241, 190.028, 192.06, 192.053
, 193.464, 195.403, 196.112, 196.502, 198.039, 199.756, 201.56, 202.79, 203.436,
 204.417, 206.468, 206.267, 208.054, 209.078, 211.095, 212.881, 213.808, 214.536
, 215.784, 217.899, 217.873, 218.163, 220.077, 221.166, 223.59, 223.031, 224.883
, 225.33, 227.874, 227.874, 228.377, 230.107, 231.005, 232.79, 233.934, 234.575,
 234.485, 235.942, 238.833, 238.401, 239.619, 240.271, 241.306, 243.892, 243.366
, 244.842, 245.339, 247.313, 247.366, 249.773, 250.483, 251.25, 253.136, 252.839
, 256.145, 255.095, 256.874, 259.246, 259.406, 261.383, 260.926, 261.737, 263.36
, 263.295, 266.204, 265.685, 267.603, 268.738, 269.688, 271.041, 271.542, 272.15
6, 273.855, 273.966, 274.657, 276.146, 278.915, 280.163, 280.403, 281.247, 281.8
27, 283.064, 284.357, 285.495, 285.844, 287.018, 289.361, 290.7, 290.138, 291.91
1, 292.418, 292.972, 295.179, 295.505, 297.301, 299.269, 300.431, 301.321, 303.0
05, 303.636, 304.324, 306.974, 307.175, 309.144, 309.866, 309.632, 311.744]
"""

#CELF sur Simmons, p = 0.01, k de 1 à 200, IC
influences_CELF_IC = [9, 16, 23, 29, 35, 41, 47, 53, 58, 62, 66, 70, 74, 78, 81, 84, 87, 90, 93, 96, 99, 102, 104, 106, 108, 110, 113, 115, 117, 119, 121, 123, 125, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 155, 156, 157, 161, 163, 165, 167, 168, 169, 170, 171, 174, 175, 178, 179, 182, 183, 184, 185, 186, 188, 190, 191, 193, 195, 196, 197, 198, 199, 200, 202, 203, 205, 206, 207, 209, 210, 211, 212, 214, 216, 217, 218, 219,220, 221, 223, 224, 225, 226, 228, 230, 231, 233, 234, 235, 236, 237, 239, 240, 242, 243, 245, 246, 248, 249, 250, 251, 252, 253, 255, 256, 258, 259, 260, 261, 264, 265, 268, 269, 270, 271, 272, 275, 276, 277, 278, 280, 281, 282, 283, 285, 288, 289, 291, 292, 293, 294, 296, 297, 299, 300, 301, 302, 303,304,307,308, 309, 310, 311, 313, 314, 315, 316, 317, 320, 321, 322, 323, 324, 326, 327, 329, 330, 331, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 346, 347, 348, 349, 350,351,353,354,355,356,357,358, 359, 361, 362]

#Degre max sur Simmons, p = 0.01, k de 1 à 200, IC
influences_degremax = [7.907, 14.402, 23.135, 28.372, 33.071, 39.522, 43.691, 49.806, 54.895, 60.48, 66.13, 71.3, 75.116, 80.667, 81.945, 85.366, 91.144, 92.957, 96.009, 99.501, 103.097, 105.754, 108.321, 111.575, 114.307, 119.066, 122.339, 125.098, 127.837, 131.15, 132.936, 136.014, 137.72, 141.299, 143.469, 146.115, 149.341, 151.235, 153.032, 156.291, 159.396, 160.097, 163.335, 164.812, 166.659, 168.94, 170.456, 173.893, 175.656, 176.551, 178.406, 180.387, 183.505, 184.888, 187.784, 189.44, 190.47, 193.603, 194.303, 195.988, 197.478, 199.516, 200.609, 203.403, 205.49, 207.157, 209.063, 209.969, 212.436, 213.055, 215.146, 216.607, 218.556, 220.441, 221.38, 222.846, 224.871, 226.285, 230.333, 230.849, 231.884, 234.943, 235.606, 236.721, 239.386, 240.242, 241.107, 242.378, 244.542, 247.259, 246.84, 249.32, 251.156, 251.76, 254.151, 255.909, 256.224, 256.885, 259.165, 261.677, 262.661, 262.991, 264.14, 265.478, 267.238, 269.181, 269.627, 271.297, 273.52, 274.551, 276.7,
277.207, 279.028, 279.723, 281.715, 282.699, 284.461, 285.755, 287.356, 287.434, 288.696, 290.459, 289.957, 293.012, 294.188, 294.759, 296.773, 297.996, 298.97, 300.509, 302.062, 303.891, 304.013, 305.05, 307.021, 308.383, 309.99, 311.386,312.686, 313.874, 314.642, 315.884, 315.852, 318.309, 319.188, 320.224, 322.341, 322.397, 324.608, 325.633, 326.68, 328.814, 329.743, 330.296, 330.765, 333.003, 334.058, 335.339, 336.41, 339.166, 339.554, 340.479, 341.674, 342.743, 344.106, 345.404, 346.262, 348.157, 348.098, 349.592, 350.554, 351.713, 353.514, 354.282, 353.875, 357.167, 357.798, 358.793, 359.083, 360.991, 362.024, 362.324, 363.676, 364.61, 366.403, 366.862, 368.918, 369.465, 370.188, 371.297, 372.791, 373.478, 374.644, 376.057, 376.597, 378.496, 379.369, 380.885, 381.227, 381.327]
k = [i for i in range(1,201)]
plt.plot(k, influences_discount, label = "Degree Discount")
plt.plot(k, influences_degremax, label = "Degrés max")
plt.plot(k, influences_CELF_IC, label = "CELF")
plt.legend()
plt.show()




##Deuxième comparaison modèles
#CELF IC, Simmons, k = 200, iterations = 1000, p = 0.01
influences = [9, 17, 24, 31, 36, 41, 46, 51, 55, 59, 63, 67, 71, 75, 79, 83, 88, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 112, 114, 116, 119, 121, 123, 125, 128, 130, 132, 134, 136, 138, 141, 142, 144, 146, 148, 149, 151, 153, 154, 155, 157, 160, 161, 163, 165, 167, 169, 170, 171, 173, 174, 175, 176, 178, 179, 181, 182, 183, 184, 187, 189, 190, 191, 192, 194, 195, 196, 197, 
200, 201, 203, 204, 205, 206, 208, 209, 210, 211, 212, 214, 216, 217, 218, 219, 220, 222, 223, 224, 227, 228, 229, 231, 232, 234, 235, 237, 239, 241, 242, 244, 245, 246, 248, 249, 250,
 252, 253, 254, 255, 257, 259, 260, 261, 262, 264, 265, 266, 267, 268, 270, 271, 272, 273, 276, 277, 278, 279, 280, 281, 282, 283, 284, 286, 288, 289, 290, 292, 293, 295, 296, 297, 298
, 299, 300, 301, 302, 304, 306, 307, 308, 309, 310, 312, 313, 314, 315, 316, 317, 318, 319, 320, 322, 324, 326, 327, 328, 329, 332, 333, 334, 335, 337, 339, 340, 341, 342, 343, 344, 345, 347, 348, 349, 350, 351, 352, 353, 354, 356, 357, 358]
#CELF LT, Simmons, k = 200, iterations = 1000
influences_LT = [12.016, 20.576, 27.448, 34.876, 41.284, 47.436, 51.204, 57.56, 61.62, 67.18, 74.068, 81.148, 85.624, 89.784, 98.488, 102.136, 107.692, 113.584, 122.764, 127.396, 133.268, 136.928, 140.644, 145.096, 149.272, 153.516, 160.284, 167.592, 173.216, 178.216, 182.128, 187.728, 191.564, 195.06, 199.836, 202.332, 207.064, 210.076, 217.788, 221.948, 227.256, 230.956, 233.492, 235.816, 243.936, 246.476, 248.616, 257.952, 260.516, 263.28, 266.444, 273.792, 276.188, 278.384, 282.572, 286.904, 289.416, 292.78, 294.704, 299.492, 301.628, 303.12, 305.896, 313.088, 321.952, 323.908, 325.084, 330.4, 331.008, 335.152, 335.704, 338.356, 338.656, 343.724, 346.144, 346.5, 346.86, 355.296, 363.612, 365.964, 366.0, 367.82, 370.516, 373.428, 383.928, 385.364, 385.712, 387.616, 393.48, 395.848, 400.868, 402.5, 404.652, 411.228, 411.72, 415.932, 419.036, 422.424, 424.684, 424.984, 426.68, 426.94, 428.012, 435.016, 438.352, 439.76, 441.728, 443.148, 444.536, 445.848, 445.796, 452.764, 456.636, 457.04, 464.796, 466.748, 467.424, 468.896, 469.964, 471.168, 473.732, 480.152, 479.688, 482.356, 482.572, 489.24, 489.26, 488.844, 493.18,
 498.748, 500.124, 500.24, 503.992, 503.472, 504.064, 506.392, 509.768, 511.948,
 513.972, 513.68, 518.016, 518.692, 520.736, 525.152, 524.936, 529.688, 529.296,
 538.74, 538.712, 538.672, 539.136, 539.764, 539.656, 540.392, 549.516, 548.684,
 552.492, 553.812, 556.12, 555.748, 556.624, 555.832, 560.556, 560.568, 561.3, 564.472, 564.78, 566.768, 566.548, 572.704, 573.48, 575.732, 575.276, 577.036, 576.572, 579.044, 578.816, 581.952, 582.86, 587.504, 588.164, 588.468, 591.224, 595.804, 594.848, 593.764, 596.832, 602.688, 603.136, 603.068, 605.544, 604.688, 608.06, 609.432, 610.048, 610.836, 611.752, 615.436, 615.924, 619.996]
 
#CELF IC, Simmons, k = 400, iterations = 1000, p = 0.015
influences_CELF_0015=[61, 91, 113, 130, 143, 154, 160, 166, 175, 180, 186, 192, 197, 203, 209, 213, 216, 220, 223, 227, 230, 232, 236, 239, 241, 244, 246, 248, 251, 253, 256, 258, 264, 266, 268, 270, 272, 275, 277, 279, 281, 283, 286, 289, 291, 293, 294, 295, 297, 298, 299, 301, 303, 306, 307, 310, 311, 313, 314, 315, 317, 318, 319, 322, 323, 326,
 327, 328, 329, 331, 333, 336, 337, 340, 342, 343, 344, 345, 346, 349, 350, 351, 352, 355, 356, 358, 359, 361, 362, 365, 366, 367, 369, 370, 371, 372, 373, 376, 377, 378, 379, 380, 381, 382, 383, 385, 386, 387, 388, 389, 393, 394, 395, 396, 397, 398, 400, 403, 404, 405, 406, 407, 408, 410, 412, 413, 414, 415, 416, 418, 420, 421, 422, 425, 426, 428, 430, 430, 433, 433, 434, 434, 436, 436, 440, 440, 441, 443, 444, 445, 445, 448, 448, 450, 451, 453, 454, 456, 456, 458, 458, 461, 462, 462, 464, 466, 466, 468, 468, 471, 472, 472, 473, 473, 475, 475, 476, 479, 481, 481, 484, 484, 484, 486, 488, 489, 490, 490, 491, 493, 495, 495, 497, 498, 499, 499, 500, 
501, 505, 505, 505, 505, 508, 508, 509, 513, 513, 514, 515, 515, 515, 519, 520, 521, 522, 524, 524, 525, 526, 527, 529, 530, 531, 531, 532, 534, 536, 538, 539, 539, 542, 542, 542, 544, 546, 547, 548, 548, 549, 550, 550, 553, 553, 554, 557, 557, 558, 558, 559, 561, 563, 563, 565, 566, 566, 566, 568, 569, 570, 571, 573, 573, 574
, 575, 576, 577, 580, 580, 581, 582, 583, 586, 586, 586, 588, 588, 590, 590, 592, 592, 593, 594, 595, 597, 597, 599, 600, 601, 602, 602, 604, 605, 606, 609, 610, 610, 610, 610, 611, 612, 613, 614, 616, 617, 618, 619, 620, 620, 620, 622, 623, 624, 625, 627, 627, 628, 629, 630, 632, 632, 633, 633, 635, 635, 636, 637, 639, 639, 641, 642, 642, 643, 645, 646, 646, 646, 648, 649, 650, 650, 651, 652, 652, 654, 656, 657, 657, 658, 660, 660, 660, 661, 663, 663, 665, 665, 666, 668, 668, 670, 671, 671, 673, 673, 674, 674, 676, 677, 678, 678, 678, 679, 681, 682, 683, 684, 685, 685, 686, 687, 688, 688, 690, 691, 693, 693, 694, 695, 696, 696, 698, 699, 699, 702,
 702, 703, 703, 704, 704, 705]
#Degre max IC , k = 400, iterations = 1000, p = 0.015
influences_degremax_0015=[41, 63, 102, 113, 123, 137, 146, 159, 172, 180, 189, 192, 201, 206, 211, 211, 221, 223, 227, 229, 233, 238, 240, 246, 249, 251, 254, 257, 261, 265, 266, 270, 271, 276, 278, 279, 283, 286, 287, 289, 291, 294, 297, 296, 299, 301, 304, 306, 307, 310, 311, 312, 314, 318, 319, 322, 322, 324, 326, 328, 330, 329, 334, 333, 337, 337,
 339, 341, 342, 343, 346, 347, 349, 350, 350, 353, 354, 355, 357, 360, 361, 363, 362, 365, 368, 368, 369, 371, 371, 373, 374, 376, 376, 379, 381, 383, 383, 383, 385, 386, 388, 388, 390, 392, 393, 393, 396, 396, 398, 399, 401, 402, 401, 404, 405, 406, 408, 408, 410, 411, 411, 412, 413, 414, 415, 417, 419, 419, 422, 421, 422, 425, 425, 428, 426, 428, 429, 430, 432, 433, 434, 435, 437, 437, 438, 438, 441, 442, 444, 444, 446, 447, 447, 448, 449, 450, 452, 453, 455, 455, 457, 458, 459, 460, 461, 461, 463, 463, 465, 465, 467, 467, 468, 468, 471, 470, 470, 473, 474, 474, 475, 477, 477, 480, 479, 480, 481, 483, 482, 485, 485, 487, 487, 488, 489, 490, 491, 
492, 494, 493, 494, 496, 498, 497, 499, 500, 501, 501, 501, 502, 504, 504, 504, 507, 508, 508, 509, 510, 512, 512, 514, 514, 514, 515, 516, 517, 518, 519, 520, 520, 522, 521, 523, 524, 524, 525, 526, 527, 528, 530, 531, 531, 533, 533, 533, 534, 536, 537, 537, 539, 539, 540, 539, 542, 542, 543, 543, 544, 546, 547, 548, 548, 550
, 550, 551, 552, 552, 553, 554, 555, 556, 557, 557, 559, 559, 561, 560, 561, 562, 563, 563, 566, 566, 566, 568, 569, 570, 570, 569, 570, 572, 573, 573, 573, 574, 576, 577, 577, 579, 579, 579, 581, 581, 582, 583, 584, 584, 585, 586, 587, 588, 589, 589, 590, 591, 592, 592, 594, 593, 594, 596, 597, 597, 597, 599, 600, 600, 600, 601, 602, 604, 605, 604, 605, 606, 606, 608, 608, 610, 610, 611, 613, 614, 614, 614, 615, 616, 616, 617, 619, 619, 620, 621, 622, 621, 622, 623, 625, 624, 626, 626, 627, 628, 628, 629, 630, 631, 632, 631, 633, 633, 635, 634, 635, 636, 638, 638, 639, 640, 640, 640, 642, 643, 643, 644, 644, 645, 646, 647, 647, 649, 649, 650, 650,
 651, 652, 652, 653, 653, 655]

#Degree Discount IC , k = 400, iterations = 1000, p = 0.015
influences_discount = [38, 88, 107, 123, 135, 143, 147, 154, 165, 172, 175, 181, 187, 189, 193, 199, 205, 207, 210, 213, 217, 221, 220, 223, 229, 229, 230, 237, 239, 239, 242, 245, 245, 251, 251, 254, 254, 254, 258, 261, 262, 266, 267, 270, 271, 270, 275, 277, 279, 282, 280, 282, 287, 286, 290, 290, 292, 293, 294, 298, 300, 302, 302, 306, 305, 306,
 309, 309, 310, 314, 314, 315, 319, 318, 320, 320, 323, 324, 324, 328, 328, 329, 331, 333, 334, 336, 337, 338, 338, 343, 341, 344, 344, 346, 346, 348, 349, 349, 353, 352, 355, 355, 354, 358, 359, 361, 361, 366, 365, 369, 367, 370, 369, 373, 374, 373, 377, 376, 376, 379, 379, 383, 381, 384, 386, 385, 388, 388, 389, 391, 389, 391, 393, 395, 393, 397, 398, 398, 398, 402, 400, 404, 401, 406, 406, 408, 410, 409, 412, 410, 413, 414, 414, 416, 419, 418, 421, 424, 424, 424, 427, 427, 430, 431, 431, 430, 435, 434, 435, 437, 440, 443, 443, 445, 448, 447, 452, 453, 454, 455, 457, 460, 463, 465, 464, 468, 467, 469, 471, 473, 474, 475, 476, 480, 482, 484, 485, 
484, 486, 489, 491, 489, 493, 493, 496, 496, 498, 499, 501, 501, 504, 506, 506, 509, 510, 510, 510, 513, 513, 515, 516, 518, 518, 519, 521, 522, 522, 525, 527, 528, 529, 530, 530, 533, 533, 534, 536, 537, 540, 540, 541, 541, 543, 545, 545, 546, 548, 550, 549, 550, 552, 555, 556, 556, 556, 560, 560, 561, 564, 564, 564, 564, 566
, 567, 569, 569, 572, 573, 574, 573, 574, 575, 579, 579, 583, 582, 582, 585, 585, 585, 587, 588, 589, 591, 590, 594, 594, 594, 596, 597, 597, 598, 598, 599, 601, 601, 604, 605, 605, 605, 609, 610, 610, 611, 612, 613, 614, 615, 615, 617, 619, 620, 620, 621, 624, 625, 624, 625, 627, 628, 627, 628, 630, 631, 633, 633, 636, 636, 636, 637, 637, 639, 640, 640, 644, 643, 645, 645, 646, 646, 648, 651, 650, 650, 652, 654, 654, 655, 658, 658, 658, 659, 659, 660, 663, 663, 667, 666, 666, 667, 669, 669, 671, 673, 672, 673, 675, 675, 676, 677, 676, 680, 681, 680, 683, 684, 683, 686, 686, 688, 688, 690, 692, 693, 692, 695, 695, 696, 697, 696, 701, 699, 701, 701,
 704, 705, 707, 705, 708, 710]
k = [i for i in range(1, 401)]
plt.plot(k, influences_discount , color = "green", label = "Degree Discount")
plt.plot(k, influences_degremax_0015, color = "red", label = "Plus hauts degrés")
plt.plot(k, influences_CELF_0015, color = "blue", label = "CELF")
plt.legend()
plt.show()

##

k = [i for i in range(1,201)]
plt.plot(k, influences_IC_0015, color = 'blue', label = "Independent Cascade (p=0.015)")
plt.plot(k, influences_LT, color = 'red', label = "Linear Threshold (poids = 1/degre)")
plt.legend()
plt.show()