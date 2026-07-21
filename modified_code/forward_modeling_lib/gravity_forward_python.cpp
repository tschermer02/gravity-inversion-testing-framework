#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#ifdef _WIN32
#define DLL_EXPORT extern "C" __declspec(dllexport)
#else
#define DLL_EXPORT extern "C"
#endif

using namespace std;

/*
    Calculate one corner term of the analytical gravity response
    of a rectangular prism.
*/
float gbox1(
    float x,
    float y,
    float z,
    float p,
    float q,
    float t
)
{
    float g;      // Gravity contribution from this corner expression
    float r;      // Distance from the observation point to the corner

    float deltx;
    float delty;
    float deltz;

    // Coordinate differences between the observation point and prism corner.
    deltx = x - p;
    delty = y - q;
    deltz = z - t;

    // Euclidean distance:
    r = std::sqrt(
        (x - p) * (x - p) +
        (y - q) * (y - q) +
        (z - t) * (z - t)
    );

    /*
        Analytical gravity expression for one rectangular-prism corner.
    */
    g = 6.67 * 0.001 *
        (
            -deltx * std::log(std::fabs(r + delty))
            -delty * std::log(std::fabs(r + deltx))
            +deltz * std::atanf(
                (deltx * delty) / (deltz * r)
            )
        );

    return g;
}


/*
    Apply the finite-boundary difference in the x direction.
    The result represents the contribution between the two x faces.
*/
float gbox2(
    float x,
    float y,
    float z,
    float p1,
    float p2,
    float q,
    float t
)
{
    float dg;

    dg =
        gbox1(x, y, z, p2, q, t)
        - gbox1(x, y, z, p1, q, t);

    return dg;
}


/*
    Extend the finite-boundary calculation into the y direction.
    This produces the combined x-y boundary contribution.
*/
float gbox3(
    float x,
    float y,
    float z,
    float p1,
    float p2,
    float q1,
    float q2,
    float t
)
{
    float dg;

    dg =
        gbox2(x, y, z, p1, p2, q2, t)
        - gbox2(x, y, z, p1, p2, q1, t);

    return dg;
}


/*
    Complete the rectangular-prism gravity calculation by applying
    the final difference in the z direction.
*/
float gbox4(
    float x,
    float y,
    float z,
    float p2,
    float p1,
    float q2,
    float q1,
    float t2,
    float t1
)
{
    float dg;

    dg =
        gbox3(x, y, z, p1, p2, q1, q2, t2)
        - gbox3(x, y, z, p1, p2, q1, q2, t1);

    return dg;
}


/*
    Calculate the gravity response of one model cell relative to
    a fixed observation point.
*/
float gbox5_va(
    int deltxn,
    int deltyn,
    int deltzn,
    int xnum,
    int ynum,
    int znum,
    float dx,
    float dy,
    float dz,
    float measureh
)
{
    float dg;

    // Observation coordinates.
    float x;
    float y;
    float z;

    // Cell boundaries in x.
    float p2;
    float p1;

    // Cell boundaries in y.
    float q2;
    float q1;

    // Cell boundaries in z.
    float t2;
    float t1;

    // Place the observation point at the horizontal origin.
    x = 0;
    y = 0;

   
    //The observation point is above the first model layer.
    z = -measureh - 0.5 * dz;

    // Upper and lower x boundaries of the selected cell.
    p2 = deltxn * dx + 0.5 * dx;
    p1 = deltxn * dx - 0.5 * dx;

    // Upper and lower y boundaries of the selected cell.
    q2 = deltyn * dy + 0.5 * dy;
    q1 = deltyn * dy - 0.5 * dy;

    // Upper and lower z boundaries of the selected cell.
    t2 = deltzn * dz + 0.5 * dz;
    t1 = deltzn * dz - 0.5 * dz;

    // Calculate the complete rectangular-cell response.
    dg = gbox4(
        x,
        y,
        z,
        p2,
        p1,
        q2,
        q1,
        t2,
        t1
    );

    return dg;
}


/*
    Precompute gravity kernel values for all possible relative offsets
    between an observation location and a model cell.

    Va_a:
        Output array containing the precomputed gravity responses.

    Relative x offsets:
        1-xnum through xnum-1
        Total count = 2*xnum - 1

    Relative y offsets:
        1-ynum through ynum-1
        Total count = 2*ynum - 1

    Depth indices:
        0 through znum-1
        Total count = znum

    Therefore, Va_a must contain at least:

        (2*xnum - 1) * (2*ynum - 1) * znum

    floating-point values.
*/
DLL_EXPORT void forward_va(
    float *Va_a,
    int xnum,
    int ynum,
    int znum,
    float dx,
    float dy,
    float dz,
    float measureh
)
{
    // Flattened index into the Va_a array.
    int n = 0;

    // This variable is declared in the original code but never used.
    float temp = 0;

    {
        /*
            Run all three loops in parallel.

            collapse(3) combines the three nested loops into one larger
            iteration space for the accelerator.
        */
        for (int vax = 1 - xnum; vax < xnum; vax++)
        {
            for (int vay = 1 - ynum; vay < ynum; vay++)
            {
                for (int vaz = 0; vaz < znum; vaz++)
                {
                    /*
                        Convert the 3D relative-offset indices into one
                        flattened array index.
                    */
                    n =
                        vaz
                        + znum * (vay + ynum - 1)
                        + (vax + xnum - 1)
                          * znum
                          * (2 * ynum - 1);

                    /*
                        Store the gravity response for this relative
                        horizontal displacement and depth.
                    */
                    Va_a[n] = gbox5_va(
                        vax,
                        vay,
                        vaz,
                        xnum,
                        ynum,
                        znum,
                        dx,
                        dy,
                        dz,
                        measureh
                    );
                }
            }
        }
    }
}


/*
    Retrieve one element A[p,q] of the gravity forward-modeling matrix.

    p:
        Observation index.

    q:
        Model-cell index.

    va:
        Precomputed gravity kernel created by forward_va().

    The full matrix A is not stored explicitly.

    Instead, this function determines the relative displacement between
    observation p and model cell q, then retrieves the corresponding
    value from the precomputed kernel array va.

    This saves memory because many elements of A have the same value
    whenever their relative offsets are identical.
*/
float getAij(
    int p,
    int q,
    float *va,
    int xnum,
    int ynum,
    int znum
)
{
    int ix;
    int iy;
    int iz;

    // Flattened index into the precomputed va array.
    int ik;

    /*
        a, b, c are the model cell's 1-based x, y, and z indices.

        q is flattened assuming x changes fastest, followed by y,
        then z.
    */
    int a;
    int b;
    int c;

    /*
        m and n are the observation point's 1-based x and y indices.

        p is flattened assuming x changes fastest.
    */
    int m;
    int n;

    /*
        Recover the model cell's z index.

        There are xnum*ynum cells in each horizontal layer.
    */
    c = q / (ynum * xnum) + 1;

    // Recover the model cell's y index.
    b = (q % (ynum * xnum)) / xnum + 1;

    // Recover the model cell's x index.
    a = ((q % (xnum * ynum)) % xnum) + 1;

    // Recover the observation point's x index.
    m = p % xnum + 1;

    // Recover the observation point's y index.
    n = p / xnum + 1;

    /*
        Compute shifted relative-offset indices.

        The added xnum and ynum values move potentially negative
        relative offsets into the positive index range used by va.
    */
    ix = m - a + xnum;
    iy = n - b + ynum;

    // Depth index comes directly from the model cell's layer.
    iz = c;

    /*
        Convert the 3D kernel indices into one flattened va index.
    */
    ik =
        (ix - 1) * (2 * ynum - 1) * znum
        + (iy - 1) * znum
        + iz - 1;

    return va[ik];
}


/*
    Perform the forward gravity calculation:

        abn_a = A * S_a

    where:

        A:
            Gravity forward-modeling matrix.

        S_a:
            Flattened 3D density model.

        abn_a:
            Predicted gravity anomaly at each surface observation point.

    Matrix dimensions:

        Arows = xnum * ynum
            Number of surface observations.

        Acols = xnum * ynum * znum
            Number of model cells.

        A therefore has shape:

            [Arows, Acols]

    The matrix A is not explicitly stored. Each A[i,m] value is
    generated by getAij() using the precomputed kernel Va_a.
*/
DLL_EXPORT void AplusS(
    float *Va_a,
    float *S_a,
    float *abn_a,
    int xnum,
    int ynum,
    int znum
)
{
    // Number of cells in the 3D density model.
    int Acols = xnum * ynum * znum;

    // Number of gravity observations on the x-y surface grid.
    int Arows = xnum * ynum;

    {
        /*
            Each observation row can be calculated independently,
            so the outer loop is parallelized.
        */
        for (int i = 0; i < Arows; i++)
        {
            // Accumulates the predicted anomaly for observation i.
            float value = 0;

            /*
                Compute the dot product between row i of A and the
                density-model vector S_a.
            */
            for (int m = 0; m < Acols; m++)
            {
                value =
                    value
                    + getAij(
                        i,
                        m,
                        Va_a,
                        xnum,
                        ynum,
                        znum
                    ) * S_a[m];
            }

            // Store the calculated gravity anomaly.
            abn_a[i] = value;
        }
    }
}


/*
    Simple test function for checking whether the compiled shared
    library can be called successfully from another language.
*/
DLL_EXPORT int test_function(int a)
{
    return 2 * a;
}


