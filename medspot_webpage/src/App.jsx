import './App.css'
import gaashlogo from './assets/gaashlogo.png'
import arxiv from './assets/arxiv.png'
import github from './assets/github.png'
import pipeline from './assets/medspot_pipeline.png'
import comparison from './assets/comparison.png'
import annotation from './assets/annotation.png'
import softwares from './assets/softwareTable.png'
import sse from './assets/sse.png'
import psw from './assets/psw.png'
import radar_comparison from './assets/radar_comparison.png'
import Failure from './failure'
import Dataexamples from './Dataexamples'



function App() {
  return (
    <span>
      <span className='flex flex-col justify-center px-7 py-7 md:20'>
        <a href='https://github.com/Tajamul21/MedSPOT' target='_blank' rel='noopener noreferrer'><img src={gaashlogo} alt="gaash logo" className='w-12 h-12 md:w-16 md:h-16'/></a>
      </span>
      
      <span className='flex flex-col items-center justify-center px-8 py-10 md:pt-4 md:px-52'>
          <h1 className='text-3xl md:text-5xl font-extrabold text-gray-700'><b>MedSPOT</b></h1>

        <h2 className='text-xl md:text-3xl text-center font-bold text-gray-600 mt-4 max-w-2xl'>
          A Workflow-Aware Sequential Grounding Benchmark for Clinical GUI
        </h2>

        {/* Add the author name etc here like mmmu */}

        <span className='flex flex-wrap justify-center pt-10'>
          <a href='https://github.com/Tajamul21/MedSPOT' target='_blank' rel='noopener noreferrer'>
            <button className='flex flex-row items-center gap-2 px-4 h-9 bg-gray-700 rounded-2xl text-white'>
              <img src={arxiv} alt="arxiv logo" className='w-3 h-3 md:w-5 md:h-5'/>
              <p className='text-xs md:text-sm text-white'>arXiv</p>
            </button>
          </a>
          <a href='https://huggingface.co/datasets/Tajamul21/MedSPOT' target='_blank' rel='noopener noreferrer'>
            <button className='flex flex-row items-center gap-2 px-4 h-9 bg-gray-700 rounded-2xl text-white'>
              <p>🤗</p>
              <p className='text-xs md:text-sm text-white'>Dataset</p>
            </button>
          </a>
          <a href='https://github.com/Tajamul21/MedSPOT' target='_blank' rel='noopener noreferrer'>
            <button className='flex flex-row items-center gap-2 px-4 h-9 bg-gray-700 rounded-2xl text-white'>
              <img src={github} alt="github logo" className='w-3 h-3 md:w-5 md:h-5'/>
              <p className='text-xs md:text-sm text-white'>Code</p>
            </button>
          </a>
          <a href='https://github.com/Tajamul21/MedSPOT' target='_blank' rel='noopener noreferrer'>
            <button className='flex flex-row items-center gap-2 px-4 h-9 bg-gray-700 rounded-2xl text-white'>
              <img src={gaashlogo} alt="gaash logo" className='w-3 h-3 md:w-5 md:h-5'/>
              <p className='text-xs md:text-sm text-white'>Gaash-Lab</p>
            </button>
          </a>
        </span>
        <span className='flex flex-row items-center'>
          <h4 className='text-2xl md:text-4xl pt-16 font-medium text-gray-600 pb-6'>Introduction</h4>
        </span>
        <span>
          <p className='text-gray-500 leading-relaxed text-justify'>We introduce MedSPOT, a workflow-aware sequential grounding benchmark for clinical GUI environments. Unlike prior benchmarks that treat grounding as a standalone prediction task, MedSPOT models procedural interaction as a sequence of structured spatial decisions. The benchmark comprises 216 task-driven videos with 597 annotated keyframes, in which each task comprises 2--3 interdependent grounding steps within realistic medical workflows. This design captures interface hierarchies, contextual dependencies, and fine-grained spatial precision under evolving conditions. To evaluate procedural robustness, we propose a strict sequential evaluation protocol that terminates task assessment upon the first incorrect grounding prediction, explicitly measuring error propagation in multi-step workflows. We further introduce a comprehensive failure taxonomy, including edge bias, small-target errors, no prediction, near miss, far miss, and toolbar confusion, to enable systematic diagnosis of model behavior in clinical GUI settings. By shifting evaluation from isolated grounding to workflow-aware sequential reasoning, MedSPOT establishes a realistic and safety-critical benchmark for assessing multimodal models in medical software environments.</p>
        </span>
        <span className='flex flex-wrap justify-center pt-10 mb-20'>
          <img src={pipeline} alt="dataset pipeline" className='w-full md:max-w-4xl h-auto'/>
        </span>

        <span className='w-screen bg-gray-100 py-8 -mx-6 md:-mx-20 '>
          <span className='flex flex-row items-center justify-center gap-2'>
            <h1 className='text-2xl md:text-4xl font-semibold text-gray-700'><b>MedSPOT</b><i>Benchmark</i></h1>
          </span>
        </span>

        <span className='flex flex-row items-center'>
          <h3 className='text-xl md:text-2xl pt-7 font-medium text-gray-600 pb-3'>Problem Formulation</h3>
        </span>
        <span>
          <p className='text-gray-500 leading-relaxed text-justify'>
            We formulate workflow-aware GUI grounding in clinical software as a sequential, 
            instruction-conditioned spatial localization problem. Unlike conventional grounding 
            benchmarks that evaluate isolated predictions, MedSPOT models grounding as a 
            temporally dependent sequence of spatial decisions within evolving interface states. 
            Steps within each task are interdependent — an incorrect grounding at any step 
            invalidates downstream actions, reflecting the procedural dependency structure 
            inherent in clinical workflows.
          </p>
        </span>
        <span className='flex flex-col items-center'>
          <h3 className='text-xl md:text-2xl pt-7 font-medium text-gray-600 pb-3'>Comparison with other benchmarks</h3>
          <p className='text-gray-500 leading-relaxed text-justify'>
            To further distinguish MedSPOT from existing GUI grounding benchmarks, we elaborate the benchmark details in the comparison figure above. From the breadth perspective, prior benchmarks are heavily focused on general-purpose desktop or web applications with isolated, single-step grounding tasks. The software environments covered are also limited in diversity. MedSPOT aims to cover clinical imaging workflows across 10 open-source medical software platforms, spanning DICOM/PACS viewers, segmentation tools, and web-based viewers — supporting 5 imaging modalities including CT, MRI, PET, X-ray, and Ultrasound. From the depth perspective, previous benchmarks typically evaluate models on independent instruction-frame pairs with no temporal dependency between steps. In contrast, MedSPOT requires workflow-aware sequential reasoning across causally interdependent steps, where an incorrect grounding at any step terminates the task — reflecting the procedural complexity and safety-critical nature of real clinical GUI environments.
          </p>
        </span>
        <span>
          <img src={comparison} alt="comparison" className='w-full md:max-w-4xl h-auto mt-5'/>
        </span>
        <span className='flex flex-col items-center'>
          <h3 className='text-xl md:text-2xl pt-7 font-medium text-gray-600 pb-3'>Annotation Protocol</h3>
          <p className='text-gray-500 leading-relaxed text-justify'>
            Real GUI interaction workflows were recorded across all 10 platforms and segmented 
            into causally consistent decision frames. Each frame was manually annotated using 
            <b> Label Studio</b> with a natural language instruction, semantic target description, normalized bounding box, and action type.
          </p>
          <img src={annotation} alt='annotation' className='pt-8 w-full md:max-w-4xl h-auto'/>
        </span>
        <span className='flex flex-col items-center'>
          <h3 className='text-xl md:text-2xl pt-7 font-medium text-gray-600 pb-3'>Domain Coverage</h3>
          <p className='text-gray-500 leading-relaxed text-justify'>
            MedSPOT spans <b>10 open-source medical imaging platforms</b> across three interface categories — DICOM/PACS viewers, segmentation and research tools, and web-based viewers.The benchmark supports <b>5 imaging modalities</b> including CT, MRI, PET, X-ray, and Ultrasound, capturing the heterogeneous GUI layouts and interaction paradigms encountered in real clinical environments.
          </p>
        </span>
        {/* Dataset Statistics */}
        <span className='flex flex-row justify-center pt-6 mb-6 gap-5'>
          <img src={softwares} alt='software details' className='pt-5 w-full md:max-w-2xl h-auto'/>
        </span>
        <span className='grid grid-cols-2 md:grid-cols-5 gap-4 w-full max-w-4xl pt-4 mb-6'>
          <span className='flex flex-col items-center bg-gray-100 rounded-2xl p-6'>
            <h4 className='text-3xl font-bold text-gray-700'>216</h4>
            <p className='text-sm text-gray-500 mt-1 text-center'>Video Tasks</p>
          </span>
          <span className='flex flex-col items-center bg-gray-100 rounded-2xl p-6'>
            <h4 className='text-3xl font-bold text-gray-700'>597</h4>
            <p className='text-sm text-gray-500 mt-1 text-center'>Annotated Keyframes</p>
          </span>
          <span className='flex flex-col items-center bg-gray-100 rounded-2xl p-6'>
            <h4 className='text-3xl font-bold text-gray-700'>10</h4>
            <p className='text-sm text-gray-500 mt-1 text-center'>Medical Platforms</p>
          </span>
          <span className='flex flex-col items-center bg-gray-100 rounded-2xl p-6'>
            <h4 className='text-3xl font-bold text-gray-700'>2-3</h4>
            <p className='text-sm text-gray-500 mt-1 text-center'>Steps per Task</p>
          </span>
          <span className='flex flex-col items-center bg-gray-100 rounded-2xl p-6'>
            <h4 className='text-3xl font-bold text-gray-700'>5</h4>
            <p className='text-sm text-gray-500 mt-1 text-center'>Modalities</p>
          </span>
        </span>
        
        <span className='flex flex-col items-center'>
          <h3 className='text-xl md:text-2xl pt-7 font-medium text-gray-600 pb-3'>Evaluation Pipeline</h3>
          <p className='text-gray-500 leading-relaxed text-justify'>
            MedSPOT follows a <b>strict sequential evaluation protocol</b>. Given a GUI frame and 
            a natural language instruction, a model predicts click coordinates. A prediction is 
            considered correct if the predicted point falls within the ground-truth bounding box. 
            Evaluation terminates upon the <b>first incorrect prediction</b> — a task is considered 
            complete only if all steps are predicted correctly in order. This explicitly measures 
            error propagation across multi-step workflows, penalizing early failures and emphasizing 
            temporal consistency.
          </p>
        </span>
        <span className='flex flex-row items center'>
          <h4 className='text-lg md:text-xl pt-7 font-medium text-gray-600 pb-3'>Metrics</h4>
        </span>
        <span className='grid grid-cols-1 md:grid-cols-4 gap-4 w-full max-w-6xl pt-2 mb-10'>
          <span className='flex flex-col items-center bg-gray-100 rounded-2xl p-6'>
            <h4 className='text-lg font-semibold text-gray-700'>TCA</h4>
            <p className='text-xs text-gray-400 mb-2'>Task Completion Accuracy</p>
            <p className='text-gray-500 text-sm text-center'>Fraction of tasks where all steps are completed correctly in sequence</p>
          </span>
          <span className='flex flex-col items-center bg-gray-100 rounded-2xl p-6'>
            <h4 className='text-lg font-semibold text-gray-700'>S1A</h4>
            <p className='test-xs text-gray-400 mb-2'>Step 1 Accuracy</p>
            <p className='text-sm text-gray-500 text-center'>Accuracy on the first step of each task</p>
          </span>
          <span className='flex flex-col items-center bg-gray-100 rounded-2xl p-6'>
            <h4 className='text-lg font-semibold text-gray-700'>SHA</h4>
            <p className='test-xs text-gray-400 mb-2'>Step Hit Accuracy</p>
            <p className='text-sm text-gray-500 text-center'>Per-step grounding accuracy across all evaluated steps.</p>
          </span>
          <span className='flex flex-col items-center bg-gray-100 rounded-2xl p-4'>
            <h4 className='text-lg font-semibold text-gray-700'>WPS</h4>
            <p className='test-xs text-gray-400 mb-2'>Weighted Prefix Score</p>
            <p className='text-sm text-gray-500 text-center '>This score reflects the disproportionate impact of early errors on overall workflow reliability.</p>
          </span>
        </span>
        {/* Failure taxonomy */}
        <span className='flex flex-col items-center'>
          <h3 className='text-xl md:text-2xl pt-7 font-light text-gray-600 pb-3'>Failure Taxonomy</h3>
          <p className='text-gray-500 leading-relaxed text-justify'>
            We introduce a comprehensive failure taxonomy to enable systematic diagnosis of model 
            behavior in clinical GUI settings, covering <b>6 failure types</b>: edge bias, 
            small-target errors, no prediction, near miss, far miss, and toolbar confusion.
          </p>
          <Failure />
        </span>
        {/* Dataexamples */}
        <span className='flex flex-col items-center'>
          <h3 className='text-xl md:text-2xl pt-10 font-light text-gray-600 pb-3'>Task Examples</h3>
          <p className='text-gray-500 leading-relaxed text-justify'>
            Each task in MedSPOT consists of a sequence of GUI interaction steps, where every step 
            includes a screenshot, a natural language instruction, and a ground-truth bounding box 
            indicating the target UI element. Below are some example tasks from different medical imaging platforms.
          </p>
          <Dataexamples />
        </span>

        {/* Results */}
        <span className='w-screen bg-gray-100 py-8 -mx-6 md:-mx-20 mt-10'>
          <span className='flex flex-row items-center justify-center'>
            <h1 className='text-2xl md:text-4xl font-semibold text-gray-700'>Evaluation Results</h1>
          </span>
        </span>
        <span className='flex flex-col items-center mb-8'>
          <h3 className='text-xl md:text-2xl pt-7 font-medium text-gray-600 pb-3'>Comparison Table</h3>
          <p className='text-gray-500 leading-relaxed'>
            Evaluation of 16 state-of-the-art MLLMs on MedSPOT. Models are ranked by Task Completion Accuracy (TCA).
          </p>
        </span>

        <div className='w-full overflow-x-auto'>
      <table className='w-full text-sm text-gray-600 border-collapse'>
          <thead>
            <tr className='bg-gray-100 text-gray-700'>
              <th className='px-4 py-3 text-left border-b border-gray-300'>Model</th>
              <th className='px-4 py-3 text-left border-b border-gray-300'>Params</th>
              <th className='px-4 py-3 text-center border-b border-gray-300'>SHR(%) ↑</th>
              <th className='px-4 py-3 text-center border-b border-gray-300'>S1A(%) ↑</th>
              <th className='px-4 py-3 text-center border-b border-gray-300'>WPS ↑</th>
              <th className='px-4 py-3 text-center border-b border-gray-300'>TCA(%) ↑</th>
            </tr>
          </thead>
          <tbody>
            {/* Closed Source */}
            <tr className='bg-blue-50'>
              <td colSpan={6} className='px-4 py-2 font-semibold text-gray-600 border-b border-gray-200'>Closed Source Models</td>
            </tr>
            {[
              { model: 'GPT 4o mini', params: '-',   shr: '4.91',  s1a: '5.14',  wps: '0.051', tca: '0.0' },
              { model: 'GPT 5',       params: '-',   shr: '16.4',  s1a: '16.5',  wps: '0.185', tca: '2.8' },
            ].map((row, i) => (
              <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                <td className='px-4 py-3 font-medium border-b border-gray-200'>{row.model}</td>
                <td className='px-4 py-3 text-center border-b border-gray-200'>{row.params}</td>
                <td className='px-4 py-3 text-center border-b border-gray-200'>{row.shr}</td>
                <td className='px-4 py-3 text-center border-b border-gray-200'>{row.s1a}</td>
                <td className='px-4 py-3 text-center border-b border-gray-200'>{row.wps}</td>
                <td className='px-4 py-3 text-center border-b border-gray-200'>{row.tca}</td>
              </tr>
            ))}

            {/* Open Source */}
            <tr className='bg-blue-50'>
              <td colSpan={6} className='px-4 py-2 font-semibold text-gray-600 border-b border-gray-200'>Open Source Models</td>
            </tr>
            {[
              { model: 'Llama 3.2 Vision-Instruct', params: '11B', shr: '0.0',   s1a: '0.0',   wps: '0.0',   tca: '0.0'  },
              { model: 'Qwen2-VL-Instruct',         params: '7B',  shr: '1.83',  s1a: '1.87',  wps: '0.02',  tca: '0.0'  },
              { model: 'DeepSeek-VL2',              params: '16B', shr: '2.7',   s1a: '2.8',   wps: '0.03',  tca: '0.0'  },
              { model: 'Gemma 3',                   params: '27B', shr: '3.7',   s1a: '1.3',   wps: '0.04',  tca: '0.0'  },
              { model: 'Mistral-3-Instruct',        params: '8B',  shr: '6.14',  s1a: '6.0',   wps: '0.064', tca: '0.0'  },
              { model: 'UGround-V1',                params: '7B',  shr: '10.17', s1a: '8.88',  wps: '0.107', tca: '0.93' },
              { model: 'Seeclick',                  params: '-',   shr: '4.3',   s1a: '9.3',   wps: '0.11',  tca: '1.4'  },
              { model: 'Qwen2.5-VL-Instruct',       params: '7B',  shr: '19.3',  s1a: '33.17', wps: '0.48',  tca: '12.6' },
              { model: 'CogAgent',                  params: '9B',  shr: '37.8',  s1a: '27.1',  wps: '0.45',  tca: '15.4' },
              { model: 'Aguvis',                    params: '7B',  shr: '54.8',  s1a: '44.0',  wps: '0.75',  tca: '26.7' },
              { model: 'Os-Atlas',                  params: '7B',  shr: '55.0',  s1a: '45.0',  wps: '0.8',   tca: '26.6' },
              { model: 'UI-Tars 1.5-VL',            params: '7B',  shr: '66.0',  s1a: '57.5',  wps: '1.0',   tca: '30.8' },
              { model: 'Qwen3-VL-Instruct',         params: '8B',  shr: '46.6',  s1a: '63.0',  wps: '1.1',   tca: '35.0' },
              { model: 'GUI-Actor',                 params: '7B',  shr: '49.6',  s1a: '65.0',  wps: '1.2',   tca: '43.5' },
            ].map((row, i) => (
              <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                <td className='px-4 py-3 font-medium border-b border-gray-200'>{row.model}</td>
                <td className='px-4 py-3 text-center border-b border-gray-200'>{row.params}</td>
                <td className='px-4 py-3 text-center border-b border-gray-200'>{row.shr}</td>
                <td className='px-4 py-3 text-center border-b border-gray-200'>{row.s1a}</td>
                <td className='px-4 py-3 text-center border-b border-gray-200'>{row.wps}</td>
                <td className='px-4 py-3 font-semibold text-center border-b border-gray-200'>{row.tca}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
        <span className='flex flex-col items-center mb-6'>
          <img src={radar_comparison} alt='comparison Radar' className='pt-6 w-full md:max-w-4xl h-auto mb-8'/>
            <p className='text-gray-500 leading-relaxed text-justify'> Above is the comparison of representative models across Step-1 Accuracy (S1A), Step Hit Rate (SHR), and Task Completion Accuracy (TCA). The consistent gap between S1A and TCA highlights compounding sequential errors under strict workflow constraints.
          </p>
        </span>

        <span className='flex flex-col items-center mb-6'>
          <h4 className='text-xl md:text-2xl pt-7 font-medium text-gray-600 pb-3'>Failure Taxonomy Results</h4>
          <p className='text-gray-500 leading-relaxed text-justify'>
            Beyond standard accuracy metrics, MedSPOT introduces a structured <b>failure taxonomy</b> 
            to enable fine-grained diagnosis of model behavior in clinical GUI settings. Each incorrect 
            prediction is categorized into one of six failure types — <b>Edge Bias</b>, <b>Far Miss</b>, 
            <b>Toolbar Confusion</b>, <b>Near Miss</b>, <b>No Prediction</b>, and <b>Small Target</b>. 
            This breakdown reveals systematic weaknesses that aggregate metrics like TCA and SHR cannot 
            capture — for instance, models that struggle specifically with small UI targets or 
            consistently misfire toward toolbar regions. Understanding these patterns is critical for 
            developing more reliable GUI-grounded models for clinical environments.
          </p>
        </span>
        {/* Failure Taxonomy Table */}
        <div className='w-full overflow-x-auto mt-10'>
          <table className='w-full text-sm text-gray-600 border-collapse'>
            <thead>
              <tr className='bg-gray-100 text-gray-700'>
                <th className='px-4 py-3 text-left border-b border-gray-300'>Model</th>
                <th className='px-4 py-3 text-left border-b border-gray-300'>Params</th>
                <th className='px-4 py-3 text-center border-b border-gray-300'>Edge Bias</th>
                <th className='px-4 py-3 text-center border-b border-gray-300'>Far Miss</th>
                <th className='px-4 py-3 text-center border-b border-gray-300'>Toolbar Confusion</th>
                <th className='px-4 py-3 text-center border-b border-gray-300'>Near Miss</th>
                <th className='px-4 py-3 text-center border-b border-gray-300'>No Prediction</th>
                <th className='px-4 py-3 text-center border-b border-gray-300'>Small Target</th>
              </tr>
            </thead>
            <tbody>
              {/* Open Source */}
              <tr className='bg-gray-50'>
                <td colSpan={8} className='px-4 py-2 font-semibold text-gray-500 border-b border-gray-200'>Open Source Models</td>
              </tr>
              {[
                { model: 'Llama 3.2 Vision-Instruct', params: '11B', eb: 49,  fm: 7,   tc: 4,   nm: 0,  np: 134, st: 19 },
                { model: 'Qwen2-VL-Instruct',         params: '7B',  eb: 129, fm: 26,  tc: 28,  nm: 0,  np: 0,   st: 27 },
                { model: 'DeepSeek-VL2',              params: '16B', eb: 30,  fm: 152, tc: 0,   nm: 2,  np: 0,   st: 30 },
                { model: 'Gemma 3',                   params: '27B', eb: 34,  fm: 44,  tc: 110, nm: 0,  np: 0,   st: 25 },
                { model: 'Mistral-3-Instruct',        params: '8B',  eb: 109, fm: 64,  tc: 13,  nm: 2,  np: 0,   st: 26 },
                { model: 'UGround-V1',                params: '7B',  eb: 62,  fm: 66,  tc: 55,  nm: 4,  np: 0,   st: 25 },
                { model: 'Seeclick',                  params: '-',   eb: 103, fm: 41,  tc: 21,  nm: 11, np: 2,   st: 29 },
                { model: 'Qwen2.5-VL-Instruct',       params: '7B',  eb: 20,  fm: 40,  tc: 78,  nm: 10, np: 10,  st: 26 },
                { model: 'CogAgent',                  params: '9B',  eb: 57,  fm: 43,  tc: 36,  nm: 3,  np: 16,  st: 26 },
                { model: 'Aguvis',                    params: '7B',  eb: 42,  fm: 84,  tc: 31,  nm: 4,  np: 0,   st: 23 },
                { model: 'Os-Atlas',                  params: '7B',  eb: 48,  fm: 49,  tc: 28,  nm: 5,  np: 6,   st: 21 },
                { model: 'UI-Tars 1.5-VL',            params: '7B',  eb: 20,  fm: 55,  tc: 30,  nm: 0,  np: 0,   st: 22 },
                { model: 'Qwen3-VL-Instruct',         params: '8B',  eb: 16,  fm: 35,  tc: 40,  nm: 6,  np: 21,  st: 18 },
                { model: 'GUI-Actor',                 params: '7B',  eb: 26,  fm: 26,  tc: 21,  nm: 1,  np: 24,  st: 17 },
              ].map((row, i) => (
                <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className='px-4 py-3 font-medium border-b border-gray-200'>{row.model}</td>
                  <td className='px-4 py-3 text-center border-b border-gray-200'>{row.params}</td>
                  <td className='px-4 py-3 text-center border-b border-gray-200'>{row.eb}</td>
                  <td className='px-4 py-3 text-center border-b border-gray-200'>{row.fm}</td>
                  <td className='px-4 py-3 text-center border-b border-gray-200'>{row.tc}</td>
                  <td className='px-4 py-3 text-center border-b border-gray-200'>{row.nm}</td>
                  <td className='px-4 py-3 text-center border-b border-gray-200'>{row.np}</td>
                  <td className='px-4 py-3 text-center border-b border-gray-200'>{row.st}</td>
                </tr>
              ))}

              {/* Closed Source */}
              <tr className='bg-gray-50'>
                <td colSpan={8} className='px-4 py-2 font-semibold text-gray-500 border-b border-gray-200'>Closed Source Models</td>
              </tr>
              {[
                { model: 'GPT 4o mini', params: '8B-10B', eb: 62, fm: 36, tc: 88, nm: 2, np: 0, st: 25 },
                { model: 'GPT 5',       params: '-',       eb: 13, fm: 67, tc: 97, nm: 3, np: 0, st: 27 },
              ].map((row, i) => (
                <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className='px-4 py-3 font-medium border-b border-gray-200'>{row.model}</td>
                  <td className='px-4 py-3 text-center border-b border-gray-200'>{row.params}</td>
                  <td className='px-4 py-3 text-center border-b border-gray-200'>{row.eb}</td>
                  <td className='px-4 py-3 text-center border-b border-gray-200'>{row.fm}</td>
                  <td className='px-4 py-3 text-center border-b border-gray-200'>{row.tc}</td>
                  <td className='px-4 py-3 text-center border-b border-gray-200'>{row.nm}</td>
                  <td className='px-4 py-3 text-center border-b border-gray-200'>{row.np}</td>
                  <td className='px-4 py-3 text-center border-b border-gray-200'>{row.st}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {/* Sequential vs Single Step */}
        <span className='w-screen bg-gray-100 py-8 -mx-6 md:-mx-20 mt-10'>
          <span className='flex flex-row items-center justify-center'>
            <h1 className='text-2xl md:text-4xl font-semibold text-gray-700'>More Results</h1>
          </span>
        </span>

        <span className='flex flex-col items-center w-full max-w-4xl pt-8 mb-10'>
          <h3 className='text-xl md:text-2xl pt-7 font-medium text-gray-600 pb-3 '>
            Sequential VS Single Step
          </h3>
          <p className='text-gray-500 leading-relaxed text-justify mb-8'>
            We compare two evaluation protocols: <b>Sequential</b>, where a task is considered 
            complete only if all steps succeed in order under early termination, and <b>Single-Step</b>, 
            where each step is scored independently without enforcing temporal dependency. Sequential 
            evaluation (TCA) yields consistently lower scores than Single-Step Accuracy (SHR) across 
            all models. The gap between SHR and TCA reflects compounding error under multi-step 
            execution. Although some models show moderate single-step accuracy, their performance 
            drops sharply under early termination — indicating that independent step scoring 
            <b> overestimates true task-level reliability</b>.
          </p>
          <img src={sse} alt='seqVssingle' className='pt-5 w-full md:max-w-2xl h-auto'/>
        </span>
        <span className='flex flex-col items-center w-full max-w-4xl pt-8 mb-6'>
          <h3 className='text-xl md:text-2xl pt-7 font-medium text-gray-600 pb-3 '>
            Per-Software Breakdown
          </h3>
          <p className='text-gray-500 leading-relaxed text-justify'>
            The per-model TCA breakdown across all ten applications reveals notable performance 
            variation. <b>GUI-Actor</b> consistently achieves the highest or near-highest performance 
            across most platforms, peaking at <b>70%</b> on ITK-SNAP and maintaining strong results 
            on MicroDICOM (58%) and Orthanc (50%). <b>UI-TARS</b> follows a similar trend, also 
            reaching 70% on ITK-SNAP. <b>AGUVIS</b> achieves the single highest score (<b>80%</b>) 
            on ITK-SNAP but exhibits greater performance variance on denser interfaces such as RadiAnt and GinkgoCADx. While <b>Qwen2.5-VL</b> demonstrates moderate performance on 3D Slicer (20%) and BlueLight (25%), it collapses on DICOM-intensive tools, underscoring limited cross-interface robustness. Overall, no model exhibits uniform generalization across all software categories — reinforcing the importance of benchmark diversity and 
            workflow-aware evaluation.
          </p>
          <img src={psw} alt='persoftware' className='pt-5 w-full md:max-w-2xl h-auto'/>
        </span>
         

         <span className='w-screen bg-gray-100 py-8 -mx-6 md:-mx-20 mt-10'>
          <span className='flex flex-row items-center justify-center'>
            <h1 className='text-2xl md:text-4xl font-semibold text-gray-700'>Citation</h1>
          </span>
        </span>

        <span className='flex flex-col items-center w-full max-w-4xl pt-8 mb-16'>
          <p className='text-gray-500 leading-relaxed text-justify mb-6'>
            If you find MedSPOT useful in your research, please consider citing our paper:
          </p>
          <div className='w-full bg-gray-50 rounded-2xl p-6 border border-gray-200'>
            <pre className='text-sm text-gray-600 overflow-x-auto whitespace-pre-wrap'>
        {`@article{medspot2026,
          title   = {MedSPOT: A Workflow-Aware Sequential Grounding Benchmark for Clinical GUI},
          author  = {Anonymous},
          year    = {2026},
          note    = {Under review}
        }`}
            </pre>
          </div>
        </span>
      </span>
    </span>
  )
}

export default App