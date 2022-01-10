namespace CartPole.Core
 
open DiffSharp.Util 
 
type Action = | Left | Right | Nothing //Nothingはキーボード入力で遊ぶときだけ必要

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
            | Nothing -> 0.0<N>
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


//module Apparatus =
//    let init () =
//        {   x         = Random.Double() * 0.1<m> - 0.05<m>
//            dx        = Random.Double() * 0.1<m/s> - 0.05<m/s>
//            theta     = Random.Double() * 0.1 - 0.05
//            dtheta    = Random.Double() * 0.1</s>  - 0.05</s> 
//            massCart  = 1.0<kg>
//            massPole  = 0.1<kg>
//            length    = 0.5<m>  
//            g         = 9.8<m/s^2>
//            tau       = 0.02<s>
//            forceMag  = 10.0<N>  
//            elappesed = 0 
//            width     = 4.8 * 120.0
//            height    = 300.0   
//            scale     = 120.0
//        }  

open FSharp.Control.Reactive   

[<AbstractClass>]
type Environment() = 
    abstract member Steps : int
    abstract member Observations : unit -> float[]
    abstract member Reflect : Action -> float[] * float * bool
    abstract member IsDone : unit -> bool
    abstract member Elappsed : unit -> int
    abstract member Reset : unit -> unit

type Env (?steps:int) = 
    inherit Environment()

    let steps = defaultArg steps 200 

    let initState () = 
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
            scale     = 120.0 } 
    let subject = initState () |> Subject.behavior 

    member _.Subject = subject

    override _.Steps = steps

    override _.Observations() = subject.Value.Observations()
    
    override _.Elappsed() = subject.Value.elappesed

    override _.Reset() = 
        subject.OnNext <| initState ()
    
    member _.Failed () =  
        let theta = subject.Value.theta
        let x = subject.Value.x 
        theta < -0.21 
        || 0.21 < theta 
        || float x < -2.4
        || 2.4 < float x     

    override this.IsDone() = 
        this.Failed() || this.Elappsed() >= steps

    override this.Reflect(action:Action)  =  

        subject.OnNext <| subject.Value.Move action 

        let reward =  
            if this.IsDone() then  
                if subject.Value.Failed() then -1.0 else 1.0 
            else 
                0.0    

        let isDone = this.IsDone() 

        subject.Value.Observations(), reward, isDone 


type Env2 (?steps:int) = 
    inherit Environment()
    let steps = defaultArg steps 200 
     
    let initState = 
        {   x         = 0.0<m> 
            dx        = 0.0<m/s>  
            theta     = System.Math.PI 
            dtheta    = 0.0</s> 
            massCart  = 1.0<kg>
            massPole  = 0.1<kg>
            length    = 0.5<m>  
            g         = 9.8<m/s^2>
            tau       = 0.02<s>
            forceMag  = 10.0<N>  
            elappesed = 0 
            width     = 4.8 * 120.0
            height    = 300.0   
            scale     = 120.0 } 
    let subject = initState |> Subject.behavior 

    member _.Subject = subject

    override _.Steps = steps

    override _.Observations() = subject.Value.Observations()
    
    override _.Elappsed() = subject.Value.elappesed

    override _.Reset() = 
        subject.OnNext <| initState
    
    member _.Failed () =  
        let x = subject.Value.x 
        float x < -2.4 || 2.4 < float x     

    override this.IsDone() = 
        this.Failed() || this.Elappsed() >= steps

    override this.Reflect(action:Action)  =  

        subject.OnNext <| subject.Value.Move action 

        let reward =  
            if this.IsDone() then   
                if subject.Value.Failed() then -1.0 
                else 
                    let theta = subject.Value.theta 
                    //if -0.21 <= theta && theta <= 0.21 then
                    //    2.0  
                    //elif -System.Math.PI / 3.0 <= theta && theta <= System.Math.PI / 3.0 then
                    //    1.0
                    //else
                    //    0.0    
                    if -System.Math.PI / 3.0 <= theta && theta <= System.Math.PI / 3.0 then
                        1.0
                    else
                        0.0 
            else 
                let theta = subject.Value.theta 
                //if -0.21 <= theta && theta <= 0.21 then
                //    2.0  
                //el
                if -System.Math.PI / 3.0 <= theta && theta <= System.Math.PI / 3.0 then
                    1.0
                else
                    0.0    

        let isDone = this.IsDone() 

        subject.Value.Observations(), reward, isDone 

open System.Net
open System.Net.Sockets
open System.Text
open System.Text.RegularExpressions
open System.Threading 
open FSharp.Control  
 

type GymCmd =  
    | Reset
    | Act of string

type GymEnvironment(client:Socket, ?steps:int)=
    inherit Environment() 
    let steps = defaultArg steps 200  
    let regexObs = Regex("^o:(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+)$", RegexOptions.Compiled) 
    let regexAct = Regex("^r:(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+)$", RegexOptions.Compiled)   
    let (|Observations|ActionResult|)  (input:string) = 
        match input[0] with
        | 'o' -> Observations (
            regexObs.Match(input).Groups 
            |> Seq.toArrayQuick |> fun ary -> ary[1..] 
            |> Array.map(fun t -> t.ToString() |> System.Double.Parse) )  
        | 'r' -> ActionResult (
            regexAct.Match(input) .Groups 
            |> Seq.toArrayQuick |> fun ary -> ary[1..] 
            |> Array.map(fun t -> t.ToString() |> System.Double.Parse)
            |> fun ary -> ary[0..3], ary[4..]
            )    
    
    let mutable elappsed = 0
    let mutable recievedAR = false
    let observations = Subject.behavior <| Array.create 4 0.0   
    let actionResult = Subject.behavior <| (0., false)
    let gymSubject = Subject.behavior GymCmd.Reset

    do   
        let buffer = Array.create 128 (byte 0)  
        gymSubject
        |> Observable.subscribe(fun cmd ->  
            match cmd with  
            | Reset -> "reset" 
            | Act a -> a  
            |> Encoding.ASCII.GetBytes |> client.Send  |> ignore
            let len = client.Receive(buffer) 
            Encoding.ASCII.GetString(buffer, 0, len) 
            |> function  
                | Observations s -> observations.OnNext s |> ignore
                | ActionResult (obs, res)  ->  
                    observations.OnNext obs
                    let failed = 
                        let obs = observations.Value
                        let x,theta = obs[0],obs[2] 
                        theta < -0.21 
                        || 0.21 < theta 
                        || float x < -2.4
                        || 2.4 < float x 
                    let isdone = failed || elappsed >= steps 
                    let reward =  
                        if isdone then  
                            if failed then -1.0 else 1.0 
                        else 
                            0.0    
                    let isDone = isdone
                    actionResult.OnNext (reward, isDone) 
                    recievedAR <- true 
        )
        |> ignore

    
    override _.Steps = steps

    override _.Observations() = 
    
        observations.Value 
    
    override _.Elappsed() = elappsed

    override _.Reset() = 
        elappsed <- 0 
        gymSubject.OnNext GymCmd.Reset 
    
    member this.Failed () = 
        let obs = observations.Value
        let x,theta = obs[0],obs[2] 
        theta < -0.21 
        || 0.21 < theta 
        || float x < -2.4
        || 2.4 < float x    

    override this.IsDone() =  
        this.Failed() || this.Elappsed() >= steps

    override this.Reflect(action:Action)  =  
        elappsed <- elappsed + 1  
        recievedAR <- false
        do
            match action with 
            | Left -> "0"
            | Right -> "1"
            | Nothing -> failwithf "this command is invalid in Gym"
            |> GymCmd.Act
            |> gymSubject.OnNext  
        
        let reward, isDone =  
            while recievedAR = false do
                Thread.Sleep 1// ActionResultの受け取りを待機 
            actionResult.Value
        observations.Value,  reward, isDone  

 