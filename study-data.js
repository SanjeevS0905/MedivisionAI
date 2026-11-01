
const studyParts = [
{
    title: "PART 1 — Introduction to Medical Imaging",
    lessons: [
        {
            id: "intro",
            title: "Introduction to Medical Imaging",
            content: [
                ["What is Medical Imaging?",
                "Medical imaging is the use of different technologies to create images of structures inside the human body. It allows healthcare professionals to examine organs, bones, tissues and other internal structures without conventional surgery."],
                ["Objectives of Medical Imaging",
                "Medical imaging supports the examination of internal structures, identification and evaluation of abnormalities, monitoring of changes over time and planning of medical care. The information obtained depends on the imaging technique used."],
                ["Major Imaging Modalities",
                "Common modalities include X-ray, computed tomography (CT), magnetic resonance imaging (MRI) and ultrasound. Each technique uses a different physical principle and provides different types of information."]
            ]
        },
        {
            id: "imaging-terms",
            title: "Basic Terms and Image Characteristics",
            content: [
                ["Resolution",
                "Resolution describes the ability of an imaging system to distinguish small structures that are close together. Higher spatial resolution can reveal finer details."],
                ["Contrast and Noise",
                "Contrast is the difference in image intensity between structures. Noise refers to unwanted variations in an image that may make structures harder to distinguish."],
                ["Pixels, Voxels and Artefacts",
                "A pixel is a two-dimensional image element. A voxel represents a small volume element in a three-dimensional image. An artefact is an image feature that does not accurately represent the structure being examined."],
                ["Region of Interest",
                "A region of interest, or ROI, is a selected area of an image used for closer examination, measurement or analysis."]
            ]
        }
    ]
},
{
    title: "PART 2 — Radiation Physics",
    lessons: [
        {
            id: "radiation-types",
            title: "Types of Radiation",
            content: [
                ["What is Radiation?",
                "Radiation is energy that travels through space or matter. It may travel as electromagnetic waves or as particles. Medical imaging uses different forms of energy to obtain information about the body."],
                ["Ionizing Radiation",
                "Ionizing radiation has enough energy to remove electrons from atoms or molecules. X-rays and gamma rays are examples. X-ray-based imaging techniques use ionizing radiation."],
                ["Non-Ionizing Radiation",
                "Non-ionizing radiation does not normally have enough energy to ionize atoms. Radiofrequency waves used in MRI and visible light are examples. Ultrasound uses mechanical sound waves rather than electromagnetic radiation."],
                ["Imaging Examples",
                "X-ray and CT use ionizing radiation. MRI uses magnetic fields and radiofrequency pulses, while ultrasound uses high-frequency sound waves."]
            ]
        },
        {
            id: "radiation-effects",
            title: "Biological Effects of Radiation",
            content: [
                ["Interaction with Biological Tissues",
                "Radiation can interact with atoms and molecules in biological tissues. The effects depend on the type of radiation, energy, amount of exposure and the tissue involved."],
                ["Direct and Indirect Effects",
                "Direct action occurs when radiation interacts with important biological molecules. Indirect action occurs when radiation interacts with other molecules, including water, producing reactive species that may affect cellular components."],
                ["Cellular Effects",
                "At sufficient exposure levels, ionizing radiation may affect DNA, chromosomes, cell membranes and proteins. Biological responses depend on exposure conditions and the ability of cells to repair damage."],
                ["Radiation Protection",
                "Medical use of ionizing radiation follows safety principles that aim to avoid unnecessary exposure while obtaining the information needed for an examination."]
            ]
        }
    ]
},
{
    title: "PART 3 — X-ray Imaging",
    lessons: [
        {
            id: "xray-principle",
            title: "Working Principle of X-ray Imaging",
            content: [
                ["Introduction",
                "X-ray imaging is a diagnostic imaging technique that uses X-rays to create images of internal body structures. X-rays are a form of high-energy electromagnetic radiation."],
                ["Production of X-rays",
                "An X-ray tube produces X-rays when high-speed electrons interact with a target material. The tube includes an electron source, a high-voltage system and a target inside an evacuated enclosure."],
                ["Differential Absorption",
                "As X-rays pass through the body, different tissues absorb or attenuate them to different extents. Dense structures such as bones generally attenuate more X-rays than many soft tissues."],
                ["Image Formation",
                "A detector records the X-rays that pass through the body. Differences in detected intensity create contrast, allowing internal structures to be represented in the final image."]
            ]
        },
        {
            id: "xray-applications",
            title: "X-ray Components and Medical Applications",
            content: [
                ["Main Components",
                "An X-ray imaging system includes an X-ray tube, high-voltage generator, collimator, patient support, detector and image-processing system."],
                ["Clinical Applications",
                "X-ray imaging is commonly used for examining bones, detecting certain chest abnormalities and supporting dental examinations. Specialised applications include mammography and fluoroscopy."],
                ["Advantages",
                "X-ray imaging is widely available, provides useful information about bones and can produce images quickly."],
                ["Limitations",
                "X-rays use ionizing radiation. Conventional X-ray images also have limitations in separating structures that overlap in a two-dimensional projection."]
            ]
        }
    ]
},
{
    title: "PART 4 — Magnetic Resonance Imaging (MRI)",
    lessons: [
        {
            id: "mri-nmr",
            title: "Nuclear Magnetic Resonance and MRI Basics",
            content: [
                ["Introduction to MRI",
                "Magnetic Resonance Imaging is a medical imaging technique that uses a strong magnetic field, radiofrequency pulses and computer processing to produce detailed images of internal body structures."],
                ["Nuclear Magnetic Resonance",
                "MRI is based on principles of nuclear magnetic resonance (NMR). Certain atomic nuclei, including hydrogen nuclei, have magnetic properties and respond to an external magnetic field."],
                ["Proton Alignment",
                "In the main magnetic field, hydrogen protons develop a net magnetisation. Radiofrequency energy at the appropriate resonance frequency changes the magnetisation, producing a signal that can be detected."],
                ["Larmor Relationship",
                "The resonance frequency is related to the strength of the magnetic field. For hydrogen, the gyromagnetic ratio is approximately 42.58 MHz/T when expressed in frequency units."]
            ]
        },
        {
            id: "mri-image",
            title: "RF Pulses, Relaxation and Image Formation",
            content: [
                ["Radiofrequency Pulses",
                "Radiofrequency pulses transfer energy to the proton system. When the pulse ends, the magnetisation changes and the system begins returning towards equilibrium."],
                ["T1 Relaxation",
                "T1 relaxation describes the recovery of longitudinal magnetisation. It is also called spin-lattice relaxation."],
                ["T2 Relaxation",
                "T2 relaxation describes the loss of phase coherence among spins in the transverse plane. It is also called spin-spin relaxation."],
                ["T2 Star and Free Induction Decay",
                "The free induction decay (FID) is the signal observed after excitation. T2* decay includes both intrinsic T2 effects and additional dephasing caused by magnetic-field inhomogeneity."],
                ["Gradients and Slice Selection",
                "Gradient coils create controlled variations in magnetic-field strength. These gradients help select slices and encode spatial information so that signals can be mapped into an image."],
                ["Image Formation",
                "The MRI system receives signals from the body and processes spatially encoded data to reconstruct images. Different acquisition settings produce different image contrasts."]
            ]
        }
    ]
},
{
    title: "PART 5 — Ultrasound Imaging",
    lessons: [
        {
            id: "ultrasound-principle",
            title: "Principles of Ultrasound and Transducers",
            content: [
                ["Introduction",
                "Ultrasound imaging uses high-frequency mechanical sound waves to examine internal body structures. It does not use ionizing electromagnetic radiation."],
                ["Basic Working Principle",
                "A transducer sends sound waves into the body. When the waves encounter boundaries between tissues, some energy is reflected back as echoes. The system processes these echoes to form an image."],
                ["Transducers",
                "A transducer converts electrical energy into sound waves and returning sound waves into electrical signals. It is the main component used to transmit and receive ultrasound."],
                ["Frequency and Penetration",
                "Higher ultrasound frequencies generally provide better detail but do not penetrate as deeply. Lower frequencies generally penetrate deeper but provide less fine detail."]
            ]
        },
        {
            id: "ultrasound-modes",
            title: "Ultrasound Modes, Image Formation and Artefacts",
            content: [
                ["Echo Formation",
                "The returning echo depends on tissue properties and the boundary between tissues. The scanner uses echo timing and strength to estimate the location and appearance of structures."],
                ["A-mode, B-mode and M-mode",
                "A-mode displays echo amplitude as peaks. B-mode produces a two-dimensional brightness image. M-mode records movement of structures over time along a selected line."],
                ["Common Artefacts",
                "Ultrasound artefacts include acoustic shadowing, posterior enhancement, reverberation, mirror-image artefacts and side-lobe artefacts. These can change how structures appear."],
                ["Applications",
                "Ultrasound is used in abdominal, cardiac, vascular, obstetric and musculoskeletal examinations. The appropriate examination depends on the clinical question."]
            ]
        }
    ]
},
{
    title: "PART 6 — Computed Tomography (CT)",
    lessons: [
        {
            id: "ct-principle",
            title: "Principles of Computed Tomography",
            content: [
                ["Introduction to CT",
                "Computed tomography uses X-rays and computer processing to create cross-sectional images of the body. Unlike a conventional X-ray projection, CT provides images representing slices through the body."],
                ["Data Acquisition",
                "During a CT examination, an X-ray tube and detector system collect measurements from multiple angles around the body."],
                ["Image Reconstruction",
                "A computer processes the measurements to reconstruct cross-sectional images. These images can also be combined to create three-dimensional representations."],
                ["Image Characteristics",
                "CT images provide information about the attenuation of X-rays by tissues. Image appearance depends on acquisition settings and reconstruction methods."]
            ]
        },
        {
            id: "ct-applications",
            title: "CT Applications, Advantages and Limitations",
            content: [
                ["Applications",
                "CT is used to examine structures such as the brain, chest, abdomen and bones. It can provide detailed information about internal anatomy."],
                ["Advantages",
                "CT can produce detailed cross-sectional images quickly and is useful for evaluating many different body regions."],
                ["Limitations",
                "CT uses ionizing radiation. Image quality and interpretation can also be affected by movement, artefacts and the examination settings."],
                ["CT and AI",
                "AI research in CT includes image reconstruction, noise reduction, organ segmentation and assistance with identifying image patterns. These applications require appropriate validation."]
            ]
        }
    ]
},
{
    title: "PART 7 — Artificial Intelligence in Medical Imaging",
    lessons: [
        {
            id: "ai-basics",
            title: "Introduction to AI, Machine Learning and Deep Learning",
            content: [
                ["What is Artificial Intelligence?",
                "Artificial intelligence refers to computer systems designed to perform tasks that normally require aspects of human intelligence, such as recognising patterns and processing information."],
                ["Machine Learning",
                "Machine learning is a branch of AI in which computer systems learn patterns from data. The learned patterns can then be used to make predictions or classify new data."],
                ["Deep Learning",
                "Deep learning uses multi-layered neural networks to learn complex patterns. Convolutional neural networks are commonly studied for image-related tasks."],
                ["Computer Vision",
                "Computer vision is the field concerned with enabling computers to analyse and interpret visual information, including medical images."]
            ]
        },
        {
            id: "ai-image-analysis",
            title: "AI Image Analysis, Segmentation and Evaluation",
            content: [
                ["Image Classification",
                "Classification assigns an image or image region to a category based on patterns learned from training data."],
                ["Object Detection",
                "Object detection identifies and locates selected objects or regions in an image, often using bounding boxes."],
                ["Image Segmentation",
                "Segmentation assigns labels to pixels or voxels to separate structures such as organs or regions of interest."],
                ["Image Enhancement",
                "AI-based enhancement methods may help reduce noise or improve image appearance. Enhancement must not be confused with creating reliable new clinical information."],
                ["Training, Validation and Testing",
                "Training data are used to teach a model. Validation data help assess and tune its performance. Test data provide a separate evaluation of how well it performs on unseen examples."],
                ["Evaluation Metrics",
                "Common evaluation measures include accuracy, precision, recall or sensitivity, specificity, F1 score and Dice coefficient. The appropriate measure depends on the task and data."]
            ]
        }
    ]
},
{
    title: "PART 8 — AI-Assisted Medical Report Understanding",
    lessons: [
        {
            id: "report-structure",
            title: "Structure and Terminology of Medical Imaging Reports",
            content: [
                ["What is a Medical Imaging Report?",
                "A medical imaging report is a written record describing an imaging examination and its observations. It communicates relevant findings to healthcare professionals."],
                ["Common Report Sections",
                "Reports may include the examination details, clinical indication, technique, findings and impression. The exact format differs between institutions and imaging modalities."],
                ["Medical Terminology",
                "Reports use specialised terms to describe anatomical structures, image appearances, measurements and observations. Understanding these terms helps readers follow the report."],
                ["Findings and Impression",
                "The findings section describes observations made during image interpretation. The impression provides a concise summary of the important conclusions."]
            ]
        },
        {
            id: "ai-report",
            title: "Natural Language Processing and AI Report Explanation",
            content: [
                ["Natural Language Processing",
                "Natural language processing (NLP) is a branch of AI that works with human language. It can be used to identify terms, abbreviations and relationships in written text."],
                ["AI-Assisted Explanation",
                "An educational report assistant can identify medical terminology and provide simpler explanations while keeping the original report available for comparison."],
                ["Summarisation",
                "AI can help produce a shorter explanation of report text. A good summary must preserve the meaning and uncertainty of the original wording rather than inventing additional findings."],
                ["Limitations and Human Review",
                "AI explanations may be incomplete or incorrect. They are not a diagnosis and cannot replace interpretation by a qualified healthcare professional. Personal medical reports should only be handled through appropriately secure and authorised systems."]
            ]
        }
    ]
},
{
    title: "PART 9 — X-ray Diffraction (XRD)",
    lessons: [
        {
            id: "xrd-principle",
            title: "Principles of X-ray Diffraction",
            content: [
                ["Introduction to XRD",
                "X-ray diffraction is a technique used to study the crystalline structure of materials. It is mainly a material-characterisation method, not a routine clinical body-imaging technique."],
                ["Diffraction",
                "When X-rays interact with the regularly arranged atoms in a crystalline material, they can produce diffraction patterns. The pattern contains information about the material's crystal structure."],
                ["Bragg's Law",
                "Bragg's law describes the condition for constructive diffraction: nλ = 2d sin θ. Here, n is an integer, λ is the X-ray wavelength, d is the spacing between crystal planes and θ is the diffraction angle."],
                ["Crystal Structure",
                "The positions and intensities of diffraction peaks help researchers study crystal spacing, phases and structural characteristics."]
            ]
        },
        {
            id: "xrd-applications",
            title: "XRD Instrumentation and Applications",
            content: [
                ["Main Instrument Components",
                "An X-ray diffractometer generally includes an X-ray source, sample holder, angular measurement system, detector and data-processing system."],
                ["Phase Identification",
                "Diffraction patterns can be compared with reference data to help identify crystalline phases in a material."],
                ["Other Applications",
                "XRD can be used to study crystal structure, residual stress and preferred crystal orientation, also known as texture."],
                ["Importance in Biotechnology",
                "XRD can support the characterisation of certain biomaterials and crystalline materials used in biological and biomedical research."]
            ]
        }
    ]
}
];
