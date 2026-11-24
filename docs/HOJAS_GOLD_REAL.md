# Hojas de Apoyo para `gold_real` (12 Abstracts)

## Caso 1/12 — `2609.37711` (cs.AR: hardware architecture)
**1. Lee una sola vez y luego oculta este párrafo:**
> In this work we look to neuromorphic computing to solve the power consumption problem that audio denoising neural networks face on edge devices like smartphones, wireless headphones and hearing aids. Spiking neural networks (SNNs) have the potential to solve this problem due to their high activation sparsity and low complexity, however many SOTA SNNs require hardware that supports a mixture of operations to be able to fully perform inference. To solve this problem, we convert SOTA audio denoising neural network Spiking-FullSubNet to a hardware friendly version showing that via QAT and activation function simplification we can achieve $\approx28\times$ improvement in power consumption to 52.9nJ per 32ms audio frame when calculated for custom digital hardware in a 45nm process node. We then propose a digital circuit which by means of a sparsity-aware flexible PE array can perform inference of the heterogeneous compute load of Spiking-FullSubNet, and validate this circuit on a PYNQ-Z1 FPGA achieving a real-time factor of 0.727 at 100MHz.

**2. Hoja de datos (mira SOLO esto al dictar o escribir en inglés):**
- **Cifras y unidades (`MEASURE`)**: 0.727, 100MHz, 32ms, 45nm, 52.9nJ
- **Términos técnicos (`TECH`)**: FPGA, FullSubNet, PE, PYNQ, QAT, SNNs, SOTA
---

## Caso 2/12 — `2610.08688` (cs.AR: hardware architecture)
**1. Lee una sola vez y luego oculta este párrafo:**
> This work presents a framework for accelerating transformer-based language models (LMs) on resource-constrained IoT devices. The framework targets compact LMs: BERT-Tiny (B-Ty), MobileBERT (M-Bt), MiniLM (M-Lm), Electra (E-Lt) and DeBERTa (D-Bt) -- selected for their architectural diversity and use in edge inference scenarios. The proposed flow derives lightweight instruction set extensions tailored to the non-obvious computational patterns of these models. In addition, a custom instruction is introduced to accelerate the address generation stage of batch matrix multiplication, achieving a performance improvement of 15.39--21.74% with modest ASIC overheads of 6.79% in area and 2.33% in power. To further enhance performance without incurring additional processor core hardware cost, an optional compiler-directed loop unrolling strategy is employed, trading increased code size for overall reduced execution time. Evaluation on the Synopsys trv32p3f RISC-V core demonstrates inference speedups of up to 2.19x, and FPGA implementation on the AMD Zynq UltraScale+ ZCU102 shows a 32.94% area overhead at 75 MHz, whereas the ASIC implementation using the TSMC 28 nm library incurs a 21.08% area overhead while operating at 250 MHz.

**2. Hoja de datos (mira SOLO esto al dictar o escribir en inglés):**
- **Cifras y unidades (`MEASURE`)**: -21.74%, 15.39, 2.19x, 2.33%, 21.08%, 250 MHz, 28 nm, 32.94%, 6.79%, 75 MHz
- **Términos técnicos (`TECH`)**: AMD, ASIC, BERT, DeBERTa, FPGA, IoT, LMs, Lm, MiniLM, MobileBERT, RISC, TSMC, ZCU102, Zynq, trv32p3f
---

## Caso 3/12 — `2611.05165` (cs.NI: networking and internet architecture)
**1. Lee una sola vez y luego oculta este párrafo:**
> During disasters, public mobile networks may become unavailable, congested, or degraded, requiring deployable communication systems to support emergency operations. This article proposes a mission-critical communications system supporing Mission Critical Push-to-Talk (MCPTT), Mission Critical Video (MCVideo), and Mission Critical Data (MCData) over a private 5G standalone network with Non-Terrestrial Network (NTN) connectivity. A context-aware Quality, Priority, and Pre-emption (QPP) mechanism is proposed to dynamically adapt network resource allocation according to the operational state of Mission Critical Services (MCX). Extensions to the Open5GS Policy Control Function (PCF) and Session Management Function (SMF) enable priority and pre-emption updates for existing flows without session re-establishment. The architecture supports tactical edge deployment of service and media functions while retaining centralized service control. Experimental evaluation considers three configurations: terrestrial backhaul with centralized media processing, satellite backhaul with centralized media processing, and satellite backhaul with edge media processing. Results show that tactical edge processing reduces communication delay and packet loss. Under satellite backhaul, edge processing reduced the 95th-percentile MCPTT audio transport delay by 60.03 ms and packet loss by 9.42 percentage points, while achieving a 95th-percentile MCPTT access time of 184.52 ms. These findings demonstrate the benefits of tactical edge media processing and provide an experimental foundation for context-aware priority mechanisms in public-safety networks integrating terrestrial and non-terrestrial components.

**2. Hoja de datos (mira SOLO esto al dictar o escribir en inglés):**
- **Cifras y unidades (`MEASURE`)**: 184.52 ms, 60.03 ms, 9.42
- **Términos técnicos (`TECH`)**: MCData, MCPTT, MCX, NTN, Open5GS, PCF, QPP, SMF
---

## Caso 4/12 — `2610.30015` (cs.NI: networking and internet architecture)
**1. Lee una sola vez y luego oculta este párrafo:**
> Antenna diversity in Internet of Things (IoT) devices is explored to address challenges associated with fixed device orientation and interference from nearby metallic objects, which can degrade wireless network performance. Integrating multiple antennas with different polarizations can significantly enhance signal quality and network reliability. Experiments demonstrated significant improvements in link quality, with gains exceeding 30 dB in received signal strength for specific scenarios. Already the addition of a second PCB-based antenna alone, resulted in a 17 dB improvement. These findings suggest that the added complexity in hardware and software to incorporate antenna diversity could well be justified by the resulting enhancements in network stability for real-world applications.

**2. Hoja de datos (mira SOLO esto al dictar o escribir en inglés):**
- **Cifras y unidades (`MEASURE`)**: 17 dB, 30 dB
- **Términos técnicos (`TECH`)**: IoT, PCB
---

## Caso 5/12 — `2611.08631` (cs.RO: robotics)
**1. Lee una sola vez y luego oculta este párrafo:**
> Camera-based driving models build a dense bird's-eye-view (BEV) grid and spend the same computation on every cell of it, although the vehicle crosses only a few of those cells within the planning horizon. We present BEVCast, a two-stage BEV whose resolution adapts to the predicted driving path. A coarse pass encodes the whole scene on a grid of 4.096 m cells. A heat head trained on the recorded path assigns a heat to every coarse cell and the cells with the highest heat are refined into 1.024 m cells under a fixed budget. The remaining cells keep their coarse resolution, so the trajectory head reads a mixed-resolution BEV that still covers the entire scene and needs neither an HD map nor a navigation command. Against a full grid of 1.024 m cells with the same architecture and training recipe, BEVCast at a 10% refinement budget encodes 28% of the BEV queries and needs 66% of the BEV encoder time at batch size 1. On nuScenes its mean displacement error is 1.078 m against 1.149 m and its drivable area violation rate is 1.46% against 2.29%. On NAVSIM navtest its mean displacement error is 0.577 m against 0.598 m and its extended predictive driver model score is 82.51 against 81.28. Next to published planners BEVCast reaches a comparable error on both benchmarks without a navigation command. The resolution of a driving BEV can therefore follow the task instead of the grid.

**2. Hoja de datos (mira SOLO esto al dictar o escribir en inglés):**
- **Cifras y unidades (`MEASURE`)**: 0.577 m, 0.598 m, 1.024 m, 1.078 m, 1.149 m, 1.46%, 10%, 2.29%, 28%, 4.096 m, 66%, 81.28, 82.51
- **Términos técnicos (`TECH`)**: BEV, HD, NAVSIM, map
---

## Caso 6/12 — `2611.12567` (cs.RO: robotics)
**1. Lee una sola vez y luego oculta este párrafo:**
> Many tasks compose transformations by a known rule: chained rotations in robotics and navigation. Learned models relearn it, compounding error; classical estimators hardcode it and miss hardware deviations. We impose the known structure and learn only the remainder: an exact non-commutative group law sits in the node of a self-contained layer, its edges learn local increments, and a zero-initialized correction psi learns unknown deviations of the law. A pre-training criterion predicts exact representability. On fixed-angle SU(2) composition, a 54-parameter dense edge is the most accurate dense model, at 27x fewer parameters than the strongest fixed-matrix baseline, and lies 0.05 deg above the Bayes floor. On real robot arms, an isolated 306-parameter psi reaches 1.12-1.14 mm versus 1.29-1.33 mm for Gauss-Newton at N >= 60, and the recipe transfers across three vendors and two protocols. The recipe matches classical fits where the data obey the law and loses where the imposed law is not the task's.

**2. Hoja de datos (mira SOLO esto al dictar o escribir en inglés):**
- **Cifras y unidades (`MEASURE`)**: 0.05 deg, 1.12, 1.14 mm, 1.29, 1.33 mm
- **Términos técnicos (`TECH`)**: SU(2
---

## Caso 7/12 — `2610.05182` (eess.IV: image and video processing)
**1. Lee una sola vez y luego oculta este párrafo:**
> Infrared image super-resolution is currently led by Mamba-based networks with 26 to 37 million parameters, which are difficult to deploy on the airborne and handheld platforms where thermal imaging is most needed. This paper presents TIRMamba, a network with 896K to 910K parameters for single-channel thermal imagery. A Thermal Prior Highway computes gradient, local-contrast and spectral cues once at the input and, through one adapter per residual group, modulates a weight-tied bidirectional state-space trunk and gates its dual-scale detail branch; a tri-path reconstruction adds the learned residual to a bicubic radiometric baseline. Because the standard benchmark provides only 265 infrared training images and evaluates fusion products on full images, we train with a replay strategy: grayscale DIV2K pre-training followed by fine-tuning on 64-pixel patches drawn with equal probability from the infrared and natural corpora. At scale factor 4, TIRMamba matches the strongest protocol-trained methods on both official test sets with 29 to 40 times fewer parameters and 2.8 to 9.4 times lower latency; at scale factor 2 it gives the highest SSIM on both. A variant with prior-conditioned selectivity, TIRMamba-Rad, corrects a 3 dB raw-thermal failure of an intermediate size-invariant design and gives the best results at scale factor 4 on raw-thermal, unmanned-aerial-vehicle and independent-sensor test sets. Code and trained models will be released at https://github.com/julian135707/TIRMamba upon acceptance.

**2. Hoja de datos (mira SOLO esto al dictar o escribir en inglés):**
- **Cifras y unidades (`MEASURE`)**: 2.8, 3 dB, 37 million, 9.4
- **Términos técnicos (`TECH`)**: DIV2, SSIM
---

## Caso 8/12 — `2610.38121` (eess.IV: image and video processing)
**1. Lee una sola vez y luego oculta este párrafo:**
> LOL-v1 and LOLv2-Real are the standard benchmarks for supervised low-light image enhancement, and several leading methods ship hand-set test-time adjustments with their evaluation scripts. We audit both benchmarks with 18 public checkpoints of eight methods under one metric implementation that reproduces every published number to within 0.09 dB, and find that their test labels differ from their training labels. (i) The 15 LOL-v1 test labels are rendered differently: the same captures appear in LOLv2 with labels of identical luminance but 1.48x lower saturation, and LOL-v1's own training labels match the LOLv2 rendering; we confirm this on the official release. (ii) The LOLv2-Real test set is the contiguous block of images 690-789, whose labels are 14% darker than the training labels (Cliff's delta = -0.45, p < 1e-12) with no comparable shift in the input statistics we examined; methods trained without any labels score up to 5.5 dB higher on test than on train. (iii) All 15 LOL-v1 test inputs reappear in LOLv2-Real, 12 in its training set. The released test-time settings are consistent with compensating for these shifts. Separating what can be calibrated on training data from what only test-set tuning finds, the test-only residual has median 0.00 dB on LOLv2-Syn, which has no shift, but 0.37 dB on LOL-v1 and 0.92 dB on LOLv2-Real (up to 3.9 dB). Under the shift, test-set checkpoint selection inflates a from-scratch model by 0.9-2.0 dB, and removing the released settings changes which method ranks first. We give concrete reporting recommendations.

**2. Hoja de datos (mira SOLO esto al dictar o escribir en inglés):**
- **Cifras y unidades (`MEASURE`)**: -0.45, 0.00 dB, 0.09 dB, 0.37 dB, 0.9, 0.92 dB, 14%, 2.0 dB, 3.9 dB, 5.5 dB
- **Términos técnicos (`TECH`)**: LOL, LOLv2
---

## Caso 9/12 — `2611.11428` (eess.SP: signal processing)
**1. Lee una sola vez y luego oculta este párrafo:**
> Wireless foundation models learn reusable representations across tasks, configurations, and deployment environments. Their pretraining requires environmental breadth, yet wireless datasets often sample few sites densely. We introduce Channels4World, a scene-centered dataset of 100,000 distinct outdoor RF scenes and a procedural method for generating them. Each scene links a generated environment and RF material assignments to 32 base-station (BS) positions, 16,384 UE positions, and aligned variable-length multipath and ordered interaction records at 3.5 and 7~GHz. The corpus comprises 52.43 billion physical BS-UE pairs, 104.86 billion frequency-specific evaluations, and a 29.14-TB packaged payload. Scene identity aligns geometry, placements, channels, and interactions across frequencies, allowing environmental breadth and within-scene observation density to vary independently. The public release offers predefined subsets at multiple scales and separate access to corpus components. A deterministic 90K/10K scene-level train-test split keeps both frequencies and all scene components together. We characterize morphology, RF materials, and transceiver placement across three generation profiles. Controlled propagation and path-retention analyses show that diffraction materially affects retained-link availability and multipath structure, while the retention policy reduces storage substantially with little loss in represented aggregate power. Under matched training exposure, models trained on broader scene collections perform better on held-out scenes, although an equal-episode control shows that scene count alone cannot explain the gain. By making environmental breadth explicit and controllable, Channels4World supports the development and evaluation of wireless foundation models for generalization across deployment environments.

**2. Hoja de datos (mira SOLO esto al dictar o escribir en inglés):**
- **Cifras y unidades (`MEASURE`)**: 104.86 billion, 16,384, 29.14, 3.5, 52.43 billion
- **Términos técnicos (`TECH`)**: BS, Channels4World, RF, TB, UE
---

## Caso 10/12 — `2611.12223` (eess.SP: signal processing)
**1. Lee una sola vez y luego oculta este párrafo:**
> Bit-interleaved coded modulation (BICM) is widely used in multiple-input multiple-output (MIMO) wireless systems. However, separating MIMO data detection from channel decoding and only exchanging per-bit soft information discards correlation information among the bits within each transmit vector. We show that the resulting information loss limits the performance of practical one-shot receivers. Even though iterative detection and decoding (IDD) can recover most of this loss, IDD significantly increases receiver complexity and latency. In order to reduce this loss, we propose COrrelation-Informed Logits (COIL), a soft-output format that combines per-bit soft information with pairwise coupling logits for correlated bits. We show how to incorporate these coupling logits into a conventional belief-propagation-based channel decoder, which enables correlation-aware decoding in a one-shot receiver without IDD. We evaluate COIL in a MIMO-BICM system with a conventional low-density parity-check code. For a 4x4 MIMO orthogonal frequency-division multiplexing (OFDM) system, a one-shot linear minimum mean-square error (LMMSE) receiver with COIL outperforms a conventional one-shot unbiased LMMSE receiver by 1.3dB and operates within 0.8dB of four-iteration IDD, while requiring only a single pass through the data detector.

**2. Hoja de datos (mira SOLO esto al dictar o escribir en inglés):**
- **Cifras y unidades (`MEASURE`)**: 0.8dB, 1.3dB
- **Términos técnicos (`TECH`)**: BICM, COIL, IDD, LMMSE, MIMO, OFDM
---

## Caso 11/12 — `2611.06053` (eess.SY: systems and control)
**1. Lee una sola vez y luego oculta este párrafo:**
> This paper presents a novel method for converting the divergent Gaussian intensity distribution of a millimeter-wave source into a uniform (flat-top) distribution using a single curved surface. The proposed method combines the surface profile of a collimator with that of a beam shaper. To validate the method, the profile of a single-sided (plano-convex) lens is derived and placed 52 mm in front of a conical horn antenna operating at 100 GHz. The lens transforms the Gaussian radiation of the horn, with an incident beam radius of 11.9 mm, into a uniform beam with a radius of 400 mm at a distance of 3 m. The lens surface profile is optimized using a full-wave-simulation-based procedure, and the resulting output beam is compared with that of a previously designed double-sided (concave-convex) lens. Simulation results show that, within the desired field of view, the optimized single-sided lens produces a flat-top intensity distribution with satisfactory ripple and roll-off. Finally, the sensitivity of the optimized lens to frequency variation is investigated. The proposed lens is suitable for generating uniform illumination in active stand-off millimeter-wave imaging systems.

**2. Hoja de datos (mira SOLO esto al dictar o escribir en inglés):**
- **Cifras y unidades (`MEASURE`)**: 100 GHz, 11.9 mm, 3 m, 400 mm, 52 mm
- **Términos técnicos (`TECH`)**: Ninguno
---

## Caso 12/12 — `2611.06050` (eess.SY: systems and control)
**1. Lee una sola vez y luego oculta este párrafo:**
> In this paper, a novel, to our knowledge, analytical approach is introduced for the design of a single-sided plano-convex lens (PCL) capable of terahertz (THz) flat-top beam shaping in the Fresnel region. The lens profile, derived from geometrical optics principles, integrates collimating and beam-shaping functions into a single optical surface, thereby simplifying fabrication. The efficiency of the method is demonstrated by designing two PCLs that convert a divergent Gaussian beam from a monochromatic THz source (at 100 and 200 GHz) into flat-top beams with target radii of 400 and 500 mm at distances of 3 and 4 m, respectively. The resulting fields, evaluated through analytical, integral equation, and finite element techniques, are shown to be comparable to those produced by a conventional double-sided concave-convex lens. Furthermore, the sensitivity of the design to frequency and alignment variations is investigated, confirming the robustness of this single-curvature approach for integration into practical THz imaging systems.

**2. Hoja de datos (mira SOLO esto al dictar o escribir en inglés):**
- **Cifras y unidades (`MEASURE`)**: 200 GHz, 4 m, 500 mm
- **Términos técnicos (`TECH`)**: PCL, PCLs
---
