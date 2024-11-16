# Scrutinising Motion Comparing Contemporary and Traditional Approaches to Optical Flow Estimation

<h2 align="left">Executive Summary</h2>

<p align="justify"> The model thoroughly examines motion and compares the performance of contemporary and traditional optical flow estimation methods. This is achieved using multiple GPUs and CPUs, parameter tuning, and diverse real-world scenarios. Furthermore, statistical analysis tests, such as confidence intervals and independent t-tests, are conducted on 30 videos to demonstrate the optimal method, thereby contributing to the field of optical flow in computer vision.</p>

<h2 align = "left">Objective</h2>

<p align="justify"> Thoroughly examine motion and compare the performance of the contemporary and traditional optical flow estimation methods.</p>

<h2 align="left">Tech Stack</h2>

- Programming Language - Python  
- Development Environment - Google Colab 
- Version Control and Collaboration - GitHub  
- Data Analytics - Pandas, NumPy, Scikit-learn  
- Machine Learning - Tensorflow  
- Data Visualisation - Matplotlib 
- Camera - Intel D435i, iphone 11
- GPU - NVIDIA RTX 3090Ti, A100, V100, T4
- CPU - Intel i9,i7, Macbook M2
- OS - Windows,Mac OS, Linux(Ubuntu)
  
<h2 align="left">Process Workflow</h2>

- <p align="justify"> Conduct research on various machine learning models and classical models to determine which one best suits the requirements,   computational resources, and time constraints</p>.

- <p align="justify">Gather the Ground Truth from Flying Chairs and MPI Sintel datasets</p>.

- <p align="justify">Capture the videos for optical flow estimation using the specified cameras.</p>

- <p align="justify">Build the framework to appropriately input data from Google Drive.</p>
- <p align="justify">Use FFmpeg for image resizing into the shape of 436 x 1024.</p>
- <p align="justify">Set up the TensorFlow architecture to handle both input and output using the videos collected and the dataset's ground truth.</p>

- <p align="justify">Build an 83-layer Deep Neural Network with 34 Convolutional 2D layers, 6 Convolution Transpose layers, 30 Activation layers, 8 Concatenation layers, 1 Correlation Cost layer, and 1 Flow Resized layer.</p>

- <p align="justify">Test with a variety of combinations of activation functions and optimisers, including Sigmoid, Swish, LeakyReLU, Adam, and SGD.</p>
- <p align="justify">Attach the custom loss function—Weighted End Point Error.</p>
- <p align="justify">Run tests for each video for 20 epochs with a batch size of 6.</p>
- <p align="justify">Conduct independent t-tests and hypothesis testing to assess performance.</p>
- <p align="justify">Record and present findings in Tableau to stakeholders.</p>

<h2 align="left">Business Value</h2>

- <p align="justify">Evaluated Performance on the the NVIDIA A100, TESLA V100 and NVIDIA RTX 3090 Ti for the machine learning model computation. Intel core i9, Intel core i7, and M2 were the CPUs under the Linux, Windows and MAC operating systems for performance evaluation</p>.

- <p align="justify">Performed statistical tests such as confidence interval and independent t-tests over a sample of 30 Videos shot through the Intel Real Sense D435i and the iPhone 11 to get confidence, difference and clarity</p>.

- <p align="justify">Analysed the parameters such as error metrics, execution Time, memory usage, temperature, the performance of activation functions, the performance of Cameras, 95 % confidence test and independent t-test visually through data visualisation tools such as Tableau.</p>

<h2 align="left" style="text-align: justify;">References</h2>

[1] S. Deqing, S. Roth and M.J. Black, "Secrets of optical flow estimation
and their principles", 2010 IEEE computer society conference on Computer
Vision and pattern recognition, pp. 2432-2439, 2010.

[2] A. Giachetti, M. Campani and V. Torre, "The use of optical flow for road
navigation," in IEEE Transactions on Robotics and Automation, vol. 14, no.
1, pp. 34-48, Feb. 1998, doi: 10.1109/70.660838.

[3] M. Khalid, L. Pénard and E. Mémin, "Application of optical flow for river
velocimetry," 2017 IEEE International Geoscience and Remote Sensing
Symposium (IGARSS), Fort Worth, TX, USA, 2017, pp. 6243-6246, doi:
10.1109/IGARSS.2017.8128436.

[4] A. H. Saputro, M. M. Mustafa, A. Hussain, O. Maskon and I. F. M. Nor,
"Myocardial motion analysis using optical flow and wavelet
decomposition," 2010 6th International Colloquium on Signal Processing
and its Applications, Malacca, Malaysia, 2010, pp. 1-5, doi:
10.1109/CSPA.2010.5545258.

[5] A. Shafaei and J. J. Little, "Real-Time Human Motion Capture with
Multiple Depth Cameras," 2016 13th Conference on Computer and Robot
Vision (CRV), Victoria, BC, Canada, 2016, pp. 24-31, doi:
10.1109/CRV.2016.25.

[6] S. Sun, Z. Kuang, L. Sheng, W. Ouyang and W. Zhang, "Optical Flow
Guided Feature: A Fast and Robust Motion Representation for Video Action
Recognition," 2018 IEEE/CVF Conference on Computer Vision and Pattern
Recognition, Salt Lake City, UT, USA, 2018, pp. 1390-1399, doi:
10.1109/CVPR.2018.00151.

[7] M. Xavier, A. Lalande, P. M. Walker, F. Brunotte and L. Legrand, "An
Adapted Optical Flow Algorithm for Robust Quantification of Cardiac Wall
Motion From Standard Cine-MR Examinations," in IEEE Transactions on
Information Technology in Biomedicine, vol. 16, no. 5, pp. 859-868, Sept.
2012, doi: 10.1109/TITB.2012.2204893.
63
[8] A. Kumar, A. Tannenbaum and G. Balas, "Optical flow: a curve evolution
approach," Proceedings., International Conference on Image Processing, Washington, DC, USA, 1995, pp. 17-20 vol.3, doi:
10.1109/ICIP.1995.537569.

[9] A. del Bimbo, P. Nesi and J. L. C. Sanz, "Optical flow computation using
extended constraints," in IEEE Transactions on Image Processing, vol. 5, no.
5, pp. 720-739, May 1996, doi: 10.1109/83.495956.

[10] A. Talukder, S. Goldberg, L. Matthies and A. Ansar, "Real-time
detection of moving objects in a dynamic scene from moving robotic
vehicles," Proceedings 2003 IEEE/RSJ International Conference on
Intelligent Robots and Systems (IROS 2003) (Cat. No.03CH37453), Las
Vegas, NV, USA, 2003, pp. 1308-1313 vol.2, doi:
10.1109/IROS.2003.1248826.

[11] -Optical flowA combined local-global approach using L1 norm Marius
Drulea1, Ioan Radu Peter2. (n.d.). In Romania Computer Science
Department, Department of Mathematics Marius.

[12] Dosovitskiy, A., Fischer, P., Ilg, E., Hausser, P., Hazirbas, C., Golkov, V.,
Smagt, P. van der, Cremers, D., and Brox, T. (2015). FlowNet: Learning
optical flow with convolutional networks. 2015 IEEE International
Conference on Computer Vision (ICCV), 2758–2766.

[13] E. Ilg, N. Mayer, T. Saikia, M. Keuper, A. Dosovitskiy and T. Brox,
"FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks,"
2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
Honolulu, HI, USA, 2017, pp. 1647-1655, doi: 10.1109/CVPR.2017.179.
[14] L. Y. Siong, S. S. Mokri, A. Hussain, N. Ibrahim and M. M. Mustafa,
"Motion detection using Lucas-Kanade algorithm and application
enhancement," 2009 International Conference on Electrical Engineering
and Informatics, Bangi, Malaysia, 2009, pp. 537-542, doi:
10.1109/ICEEI.2009.5254757.

[15] D. Zhang and G. Lu, "An edge and color oriented optical flow
estimation using block matching," WCC 2000 ICSP 2000. 2000 5th
International Conference on Signal Processing Proceedings. 16th World
Computer Congress 2000, Beijing, China, 2000, pp. 1026-1032 vol.2, doi:
10.1109/ICOSP.2000.891703.

[16] R. Pericet-Camara, G. Bahi-Vila, J. Lecoeur and D. Floreano, "Miniature
artificial compound eyes for optic-flow-based robotic navigation," 2014
64 13th Workshop on Information Optics (WIO), Neuchatel, Switzerland, 2014,
pp. 1-3, doi: 10.1109/WIO.2014.6933290.

[17] B. Zhang, "Computer vision vs. human vision," 9th IEEE International
Conference on Cognitive Informatics (ICCI'10), Beijing, China, 2010, pp. 3-3,
doi: 10.1109/COGINF.2010.5599750.

[18] K. Kale, S. Pawar and P. Dhulekar, "Moving object tracking using optical
flow and motion vector estimation," 2015 4th International Conference on
Reliability, Infocom Technologies and Optimization (ICRITO) (Trends and
Future Directions), Noida, India, 2015, pp. 1-6, doi:
10.1109/ICRITO.2015.7359323.

[19] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W.
Hubbard, and L. D. Jackel. Backpropagation applied to handwritten zip code
recognition. Neural Computation, 1(4):541–551, 1989

[20] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification
with deep convolutional neural networks. In NIPS, pages 1106–1114, 2012.

[21] R. Jonschkowski, A. Stone, J. T., Barron, A. Gordon, K. Konolige, A.
Angelova (2020). What Matters in Unsupervised Optical Flow. In: A. Vedaldi, H. Bischof, T. Brox,J. M. Frahm (eds) Computer Vision ECCV 2020.
ECCV 2020. Lecture Notes in Computer Science(), vol 12347. Springer,
Cham. https://doi.org/10.1007/978-3-030-58536-5_33

[22] D. J. Fleet and Y. Weiss (n.d.). Optical Flow Estimation. Toronto.edu.
Retrieved July 31, 2023, from
https://www.cs.toronto.edu/~fleet/research/Papers/flowChapter05.pdf

[23] T-W. Hui, X. Tang and C. C. Loy, "LiteFlowNet: A Lightweight
Convolutional Neural Network for Optical Flow Estimation," 2018 IEEE/CVF
Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT,
USA, 2018, pp. 8981-8989, doi: 10.1109/CVPR.2018.00936.

[24] A. Ranjan and M.J. Black, "Optical flow estimation using a spatial
pyramid network", CVPR, pp. 4161-4170, 2017.

[25] D. Fortun, P. Bouthemy, and C. Kervrann (2015). Optical flow modeling
and computation: a survey. Computer Vision and Image Understanding
65

[26] A. Geiger, P. Lenz, and R. Urtasun (2012). Are we ready for
autonomous driving? The KITTI vision benchmark suite. 2012 IEEE
Conference on Computer Vision and Pattern Recognition.

[27] D. J. Butler, J. Wulff, G. B. Stanley, and M. J. Black (2012). A naturalistic
open source movie for optical flow evaluation. In Computer Vision ECCV
2012 (pp. 611–625). Springer Berlin Heidelberg.

[28] S. Baker, D. Scharstein, J. Lewis, S. Roth, M. J. Black, and R. Szeliski. A
database and evaluation methodology for optical flow.In ICCV, 2007.

[29] Metrics Performance Analysis of Optical Flow Taha Alhersh1 a , Samir
Brahim Belhaouari2 b and Heiner Stuckenschmidt1 c 1Data and Web
Science Group. (n.d.). In 2College of Science and Engineering.

[30] Depth Camera D435. (2019, February 20). Intel® RealSenseTM Depth
and Tracking Cameras; Intel RealSense.
https://www.intelrealsense.com/depth-camera-d435/

[31] IPhone 11 technical specifications. (n.d.). Apple.com. Retrieved 21
August 2023, from https://support.apple.com/kb/SP804?locale=en_GB

[32] (N.d.). SINTEL.Is. Retrieved August 22, 2023, from
http://SINTEL.is/downloads

[33] L. Alzubaidi, J.Zhang, A. J. Humaidi, A. Al-Dujaili, Y. Duan, O.
Al-Shamma, J. Santamaría, M.A. Fadhel, M. Al-Amidie, and L. Farhan (2021).
Review of deep learning: concepts, CNN architectures, challenges,
applications, future directions. Journal of Big Data, 8(1).
https://doi.org/10.1186/s40537-021-00444-8

[34] A. Khan, A. Sohail, U. Zahoora, and Qureshi, A. S. (2019). A survey of
the recent architectures of deep convolutional Neural Networks. In arXiv
[cs.CV]. http://arxiv.org/abs/1901.06032

[35] B. Quoc (n.d.). Searching For Activation Functions Prajit
Ramachandran.

[36] C. Szegedy, S. Ioffe, V. Vanhoucke, A. and Alemi (2016). Inception-v4,
Inception-ResNet and the impact of residual connections on learning. In
arXiv [cs.CV]. http://arxiv.org/abs/1602.0726

[37] Lin, C.-E. (2019, April 24). Introduction to motion estimation with
Optical Flow. Nanonets AI and Machine Learning Blog.
https://nanonets.com/blog/optical-flow/66

[38] Motion Analysis and Object Tracking — OpenCV 3.0.0-dev
documentation. (n.d.). Opencv.org. Retrieved 26 August 2023, from
https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_an
d_object_tracking.html?ref=nanonets.com

[39] C. Vogel, S. Roth, K. Schindler (2013). An Evaluation of Data Costs for
Optical Flow. In: Weickert, J., Hein, M., Schiele, B. (eds) Pattern Recognition.
GCPR 2013. Lecture Notes in Computer Science, vol 8142. Springer, Berlin,
Heidelberg. https://doi.org/10.1007/978-3-642-40602-7_37

[40] B. Chiang and J. Bohg (n.d.). Optical and scene flow. Stanford.edu.
Retrieved 26 August2023,from,https://web.stanford.edu/class/cs231a/course_notes/09- optical-flow.pdf

[41] A. Plyer, G. Le Besnerais and F. Champagnat Massively parallel
Lucas-Kanade optical flow for real-time video processing applications. J Real-Time Image Proc 11, 713–730 (2016).
https://doi.org/10.1007/s11554-014-0423-0

[42] L. Guan, L. Zhai, H. Cai, P. Zhang, Y. Li, J. Chu, R. Jin, and H. Xie (2020).
Study on displacement estimation in low illumination environment through
polarized contrast-enhanced optical flow method for polarization
navigation applications. Optik, 210(164513), 164513.
https://doi.org/10.1016/j.ijleo.2020.164513

[43] R. Marsal, F. Chabot, A. Loesch, and H. Sahbi (n.d.). BrightFlow:
Brightness-change-aware unsupervised learning of optical flow.
Thecvf.com. Retrieved August 30, 2023, from
https://openaccess.thecvf.com/content/WACV2023/papers/Marsal_BrightF
low_Brightness-Change-Aware_Unsupervised_Learning_of_Optical_Flow_
WACV_2023_paper.pdf

[44] (N.d.). Researchgate.net. Retrieved August 30, 2023, from
https://www.researchgate.net/publication/221002534_Optical_flow_using
_color_information

[45] V. Sze, Y-H. Chen, J. Emer, A. Suleiman, and Z. Zhang (n.d.). Hardware for Machine Learning: Challenges and Opportunities. Arxiv.org. Retrieved
August 30, 2023, from http://arxiv.org/abs/1612.07625

[46] B. Ko, H-G. Kim, K-J. Oh and H-J. Choi, "Controlled dropout: A different
approach to using dropout on deep neural network," 2017 IEEE
International Conference on Big Data and Smart Computing (BigComp),67
Jeju, Korea (South), 2017, pp. 358-362, doi: 10.1109/BIGCOMP.2017.7881693.

[47] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R.
Salakhutdinov (n.d.). Dropout: A simple way to prevent neural networks
from overfitting. Toronto.edu. Retrieved 30 August 2023, from
https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf

[48] W. Wang, "Confidence Intervals for Ultra-High Reliability:
Computations, Comparisons and Recommendations," 2018 Annual
Reliability and Maintainability Symposium (RAMS), Reno, NV, USA, 2018,
pp. 1-5, doi: 10.1109/RAM.2018.8463069.

[49] N. Hu and E-H. Yang, "Confidence interval based motion estimation,"
2013 IEEE International Conference on Image Processing, Melbourne, VIC,
Australia, 2013, pp. 1588-1592, doi: 10.1109/ICIP.2013.6738327.

[50] R. Sanchez-Matilla and A. Cavallaro, "Confidence Intervals for Tracking
Performance Scores," 2018 25th IEEE International Conference on Image
Processing (ICIP), Athens, Greece, 2018, pp. 246-250, doi:
10.1109/ICIP.2018.8451433.

[51] D. W. Coit, "System-reliability confidence-intervals for
complex-systems with estimated component-reliability," in IEEE
Transactions on Reliability, vol. 46, no. 4, pp. 487-493, Dec. 1997, doi:
10.1109/24.693781.

[52] B. Allaert, I. R. Ward, I.M.Bilasco, C. Djeraba, and M. Bennamoun
(n.d.). A comparative study on Optical Flow for Facial Expression Analysis.
Arxiv.org. Retrieved 31 August 2023, from http://arxiv.org/abs/1904.11592

[53] https://github.com/Anshul12256/MultiFlow-Optical-Flow-Estimation-Using
-Deep-Neural-Networks

[54] J. Schmidhuber (n.d.). Annotated history of modern AI and deep
learning. Arxiv.org. Retrieved 4 September 2023, from
http://arxiv.org/abs/2212.11279

[55] H. Rashed, A. El Sallab, S. Yogamani, and M. ElHelw, “Motion and
depth augmented semantic segmentation for autonomous navigation,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition Workshops, 2019.68

[56] The KITTI Vision Benchmark Suite. (n.d.). Cvlibs.net. Retrieved 6
September 2023, from https://www.cvlibs.net/datasets/kitti/

[57] M. Au-Yong-Oliveira, C. Lopes, F. Soares, G. Pinheiro and P. Guimarães,
"What can we expect from the future? The impact of Artificial Intelligence
on Society," 2020 15th Iberian Conference on Information Systems and
Technologies (CISTI), Seville, Spain, 2020, pp. 1-6, doi:
10.23919/CISTI49556.2020.9140903.

[58] B. Marr (2023, June 15). Why companies are vastly underprepared for
the risks posed by AI. Forbes.
https://www.forbes.com/sites/bernardmarr/2023/06/15/why-companies-a
re-vastly-underprepared-for-the-risks-posed-by-ai

[59] Biswal, A. (2020, September 18). 7 types of artificial intelligence that you should know in 2023. Simplilearn.com;
Simplilearn.https://www.simplilearn.com/tutorials/artificial-intelligence-tut
orial/types-of-artificial-intelligence.

[60] TensorFlow. (n.d.). TensorFlow. Retrieved 8 September 2023, from
https://www.tensorflow.org/

[61] Tf.Keras.Optimizers.AdamW. (n.d.). TensorFlow. Retrieved 8
September2023,fromhttps://www.tensorflow.org/api_docs/python/tf/kera
s/optimizers/AdamW

[62] https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/experim
ental/SGD

[63] NumPy. (n.d.). Numpy.org. Retrieved 8 September 2023, from
https://numpy.org/

[64] Pandas documentation — pandas 2.1.0 documentation. (n.d.).
Pydata.org. Retrieved 8 September 2023, from
https://pandas.pydata.org/docs/

[65] Matplotlib — visualization with python. (n.d.). Matplotlib.org.
Retrieved 8 September 2023, from https://matplotlib.org/

[66] Keras: Deep learning for humans. (n.d.). Keras.Io. Retrieved 8
September 2023, from https://keras.io/69

[67] SciPy. (n.d.). Scipy.org. Retrieved 8 September 2023, from
https://scipy.org/

[68] Opencv-python. (n.d.). PyPI. Retrieved 8 September 2023, from
https://pypi.org/project/opencv-python/

[69] Pillow. (n.d.). PyPI. Retrieved 8 September 2023, from
https://pypi.org/project/Pillow/

[70] os — Miscellaneous operating system interfaces. (n.d.). Python
Documentation. Retrieved 8 September 2023, from https://docs.python.org/3/library/os.html

[71] time — Time access and conversions. (n.d.). Python Documentation.
Retrieved 8 September 2023, from https://docs.python.org/3/library/time.html

[72] Welcome to. (n.d.). Python.org. Retrieved 8 September 2023, from
https://www.python.org/

[73] Colab.Google. (n.d.). Colab.Google. Retrieved 8 September 2023, from
https://colab.google/

[74] L. Jiaxi, "The Application and Research of T-test in Medicine," 2010
First International Conference on Networking and Distributed Computing,
Hangzhou, China, 2010, pp. 321-323, doi: 10.1109/ICNDC.2010.70.

[75] T. Riaji, S. E. Hassani and F. E. M. Alaoui, "Application of Paired
Samples t-Test in Engineering Service-Learning Project," 2022 11th
International Symposium on Signal, Image, Video and Communications
(ISIVC), El Jadida, Morocco, 2022, pp. 1-4, doi: 10.1109/ISIVC54825.2022.9800209.

[76] K. M. Htay, R. R. Othman, A. Amir, H. L. Zakaria and N. Ramli, "A
Pairwise T-Way Test Suite Generation Strategy Using Gravitational Search
Algorithm," 2021 International Conference on Artificial Intelligence and
Computer Science Technology (ICAICST), Yogyakarta, Indonesia, 2021, pp. 7-12, doi: 10.1109/ICAICST53116.2021.9497823.

[77] K-T. Shih and H. H. Chen, "Generating High-Resolution Image and
Depth Map Using a Camera Array With Mixed Focal Lengths," in IEEE
Transactions on Computational Imaging, vol. 5, no. 1, pp. 68-81, March
2019, doi: 10.1109/TCI.2018.2871391.70

[78] S. J. Raudys and A. K. Jain, "Small sample size effects in statistical
pattern recognition: recommendations for practitioners," in IEEE
Transactions on Pattern Analysis and Machine Intelligence, vol. 13, no. 3,
pp. 252-264, March 1991, doi: 10.1109/34.75512.

[79] Independent T-Test – An introduction to when to use this test and
what are the variables required. (n.d.). Laerd.com. Retrieved 9 September
2023, from https://statistics.laerd.com/statistical-guides/independent-t-test-statistical￾guide.php

[80] R. Dasoriya, J. Rajpopat, R. Jamar and M. Maurya, "The Uncertain
Future of Artificial Intelligence," 2018 8th International Conference on
Cloud Computing, Data Science and Engineering (Confluence), Noida, India,
2018, pp. 458-461, doi: 10.1109/CONFLUENCE.2018.8442945.

[81] (N.d.-b). Pexels.com. Retrieved 11 September 2023, from
https://www.pexels.com/video/2880726/

[82] (N.d.-c). Nvidia.com. Retrieved 11 September 2023, from
https://www.nvidia.com/content/dam/en-zz/Solutions/geforce/ampere/rtx
-3090/geforce-rtx-3090-shop-600-p@2x.png

[83] (N.d.-d). Nvidia.com. Retrieved 11 September 2023, from
https://www.nvidia.com/content/dam/en-zz/es_em/es_em/Solutions/Data
-Center/tesla-v100/data-center-tesla-v100-nvlink-625-ud@2x.jpg

[84] (N.d.-e). Wikimedia.org. Retrieved 11 September 2023, from
https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.s
vg

[85] (N.d.-f). Wikimedia.org. Retrieved 11 September 2023, from
https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Tux.svg/12
00px-Tux.svg.png

[86] (N.d.-g). Wikimedia.org. Retrieved 11 September 2023, from
https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Windows_l
ogo_-_2012.svg/1024px-Windows_logo_-_2012.svg.png

[87] FFmpeg. (n.d.). Ffmpeg.org. Retrieved 16 September 2023, from
https://ffmpeg.org/

[88] What is R? (n.d.). R-project.org. Retrieved 16 September 2023, from
https://www.r-project.org/about.html 71

[89] S. A. Abdul Ameer, R. Khalid, A. H. O. Al Mansor and P. Singh, "Hybrid
Deep Neural Networks for Improved Sentiment Analysis in Social Media,"
2023 Third International Conference on Secure Cyber Computing and
Communication (ICSCCC), Jalandhar, India, 2023, pp. 542-547, doi:
10.1109/ICSCCC58608.2023.10176880.

[90] H. Zhao and L. Chen, "Artificial Intelligence Security Issues and
Responses," 2020 IEEE 6th International Conference on Computer and
Communications (ICCC), Chengdu, China, 2020, pp. 2276-2283, doi:
10.1109/ICCC51575.2020.9345035.

[91] N. Mayer, E. Ilg, P. Häusser, P. Fischer, D. Cremers, A. Dosovitskiy and T.
Brox (n.d.). FlyingThings3D Dataset [Data set].

[92] D. C. Schmidt, M. Deshpande and C. O'Ryan, "Operating system
performance in support of real-time middleware," Proceedings of the
Seventh IEEE International Workshop on Object-Oriented Real-Time
Dependable Systems. (WORDS 2002), San Diego, CA, USA, 2002, pp.
199-206, doi: 10.1109/WORDS.2002.1000053.

[93] A. S. Bansal (n.d.). No title.

[94] M. Nicely (n.d.). Overview - CUDA python 12.2.0 documentation.
Github.Io. Retrieved 25 September 2023, from
https://nvidia.github.io/cuda-python/overview.html 72
