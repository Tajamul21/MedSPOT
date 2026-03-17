import { useState } from "react";
import figure1 from './assets/figure1.png'
import figure2 from './assets/figure2.png'
import figure3 from './assets/figure3.png'
import figure4 from './assets/figure4.png'
import figure5 from './assets/figure5.png'
import figure6 from './assets/figure6.png'


const failureTypes=[
    {img: figure1, label:'Edge Bias'},
    {img: figure2, label:'Small Target'},
    {img: figure3, label:'No Prediction'},
    {img: figure4, label:'Near Miss'},
    {img: figure5, label:'Far Miss'},
    {img: figure6, label:'Toolbar Confusion'},
]
function Failure(){
    const [current, setCurrent] = useState(0);
    const prev = ()=>setCurrent((current - 1 + failureTypes.length) % failureTypes.length)
    const next = ()=>setCurrent((current + 1) % failureTypes.length)
    return(
        <span className="flex flex-col items-center w-full max-w-4xl pt-6 mb-10">
            <span className="relative flex flex-row items-center justify-center gap-4 w-full">
                <button onClick={prev} className="w-10 h-10 rounded-full bg-gray-200 hover:bg-gray-400 flex items-center justify-center text-gray-800">
                    ‹
                </button>
                <span>
                    <img src={failureTypes[current].img} alt={failureTypes[current].label} className='w-full md:max-w-2xl h-auto rounded-xl'/>
                    <p className="text-sm mt-3 text-gray-500">{failureTypes[current].label}</p>
                </span>
                <button onClick={next} className="w-10 h-10 rounded-full bg-gray-200 hover:bg-gray-400 flex items-center justify-center text-gray-800">
                    ›
                </button>

            </span>
        </span>
    )
}

export default Failure