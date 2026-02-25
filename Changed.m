
    function changed = Changed(Problem, Population)
    % Detect whether the problem changes

    % Sample a subset of the population
    RePop1 = Population(randperm(end, ceil(end/5)));

    % Re-evaluate the sampled solutions
    RePop2 = Problem.Evaluation(RePop1.decs, false);

    % Compute the absolute difference in objectives and constraints
    deltaObjs = abs(RePop1.objs - RePop2.objs);
    deltaCons = abs(RePop1.cons - RePop2.cons);

    % Determine if any changes exceed the threshold
    threshold = 1e-3;
    objsChanged = any(deltaObjs(:) > threshold);
    consChanged = any(deltaCons(:) > threshold);

    % Consider the problem as changed if any changes exceed the threshold
    changed = objsChanged || consChanged;
end