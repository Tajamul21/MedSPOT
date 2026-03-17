import { useState } from "react";
import figure7 from './assets/figure7.png'
import figure8 from './assets/figure8.png'
import figure9 from './assets/figure9.png'

const exampleTypes=[
    {img:figure7, label:'Example1'},
    {img:figure8, label:'Example2'},
    {img:figure9, label:'Example3'},
]

function Dataexamples(){
    const [current, setCurrent] = useState(0);

    const prev = ()=>setCurrent((current - 1 + exampleTypes.length) % exampleTypes.length)
    const next = ()=>setCurrent((current + 1) % exampleTypes.length)

    return (
        <span className="flex flex-col items-center w-full max-w-4xl pt-6 mb-10">
            <span className="relative flex flex-row items-center justify-center gap-4 w-full">
                <button onClick={prev} className="w-10 h-10 rounded-full bg-gray-200 hover:bg-gray-400 flex items-center justify-center text-gray-800">
                    ‹
                </button>
                <span>
                    <img src={exampleTypes[current].img} alt={exampleTypes[current].label} className='w-full md:max-w-2xl h-auto rounded-xl'/>
                    <p className="text-sm mt-3 text-gray-500">{exampleTypes[current].label}</p>
                </span>
                <button onClick={next} className="w-10 h-10 rounded-full bg-gray-200 hover:bg-gray-400 flex items-center justify-center text-gray-800">
                    ›
                </button>

            </span>
        </span>
    )
}

export default Dataexamples