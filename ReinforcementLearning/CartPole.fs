namespace CartPole
 
open DiffSharp.Util 
 
type Action = | Left | Right | None

type [<Measure>] kg 
type [<Measure>] m 
type [<Measure>] s
type [<Measure>] N = kg m / s^2 

type Apparatus = { 
    x         : float<m>
    dx        : float<m/s>
    theta     : float
    dtheta    : float</s>
    massCart  : float<kg>
    massPole  : float<kg>
    length    : float<m>
    g         : float<m/s^2>
    tau       : float<s>
    forceMag  : float<N>  
    width     : float
    height    : float   
    scale     : float
    elappesed : int 

} with 
    member this.Move action  = 
        // http://www.rel.hiroshima-u.ac.jp/inverted/index.html
        // https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        let force =
            match action with
            | Left  ->  -this.forceMag
            | Right -> this.forceMag
            | None -> 0.0<N>
        let sinTheta = this.theta |> float |> sin 
        let cosTheta = this.theta |> float |> cos    
        let poleMassLength =  this.massPole * this.length 
        let massTotal = this.massPole + this.massCart
        let tmp = (force + poleMassLength * sinTheta * this.dtheta * this.dtheta) / massTotal
        let ddtheta =
            (this.g * sinTheta - cosTheta * tmp) 
            / (this.length * (4.0 / 3.0 - this.massPole * cosTheta * cosTheta / massTotal)) 
        let ddx = tmp - poleMassLength * ddtheta * cosTheta / massTotal

        { this with
            x         = this.x + this.dx * this.tau
            dx        = this.dx + ddx * this.tau
            theta     = this.theta + this.dtheta * this.tau
            dtheta    = this.dtheta + ddtheta * this.tau  
            elappesed = this.elappesed + 1  
        } 

    member this.Observations()   =   
        [|  float this.x
            float this.dx
            float this.theta
            float this.dtheta
        |] 
    
    member this.Failed ()  = 
        this.theta < -0.21 
        || 0.21 < this.theta 
        || float this.x < -2.4
        || 2.4 < float this.x   


module Apparatus =
    let init () =
        {   x         = Random.Double() * 0.1<m> - 0.05<m>
            dx        = Random.Double() * 0.1<m/s> - 0.05<m/s>
            theta     = Random.Double() * 0.1 - 0.05
            dtheta    = Random.Double() * 0.1</s>  - 0.05</s> 
            massCart  = 1.0<kg>
            massPole  = 0.1<kg>
            length    = 0.5<m>  
            g         = 9.8<m/s^2>
            tau       = 0.02<s>
            forceMag  = 10.0<N>  
            elappesed = 0 
            width     = 4.8 * 120.0
            height    = 300.0   
            scale     = 120.0
        }  

open FSharp.Control.Reactive   

type Environment (?steps:int) = 

    let steps = defaultArg steps 200

    let subject = Subject.behavior <| Apparatus.init  ()  

    member _.Subject = subject

    member _.Steps = steps

    member _.State() = subject.Value.Observations()
    
    member _.Elappsed() = subject.Value.elappesed

    member _.Reset() = 
        subject.OnNext <| Apparatus.init ()
    
    member this.Failed () = 
        subject.Value.Failed()  

    member this.IsDone() = 
        this.Failed() || this.Elappsed() >= steps

    member this.Reflect(action:Action)  =  

        subject.OnNext <| subject.Value.Move action 

        let reward =  
            if this.IsDone() then  
                if this.Failed () then -1.0 else 1.0 
            else 
                0.0    

        let isDone = this.IsDone() 

        subject.Value.Observations(), reward, isDone

