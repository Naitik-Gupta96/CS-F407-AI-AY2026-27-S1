% Optional Logic Hands-on extension: Prolog verifier

connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).

can_move(X,Y) :-
    connected(X,Y).

valid_move(X,Y) :-
    connected(X,Y).

% Queries from the lab:
% ?- can_move(a,b).     -> true.
% ?- can_move(a,c).     -> false.
% ?- valid_move(a,b).   -> true.
% ?- valid_move(b,c).   -> true.
% ?- valid_move(a,c).   -> false.

% The lab's proposed plan:
% Move(a,b)
% Move(b,c)
%
% Both moves are supported by the connected facts.

% The challenge Move(a,c) should fail:
% ?- valid_move(a,c).
% false.

% Task 8:
wet_road.
slippery :-
    wet_road.
reduce_speed :-
    slippery.

% ?- reduce_speed.
% true.
%
% Reasoning:
% wet_road => slippery => reduce_speed
