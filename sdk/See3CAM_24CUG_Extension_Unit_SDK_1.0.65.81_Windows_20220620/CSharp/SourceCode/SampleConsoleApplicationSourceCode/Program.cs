using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.Threading;

namespace SampleAppDevicePath
{
    class Program
    {
        static unsafe void Main(string[] args)
        {
            int SelectedDevice;
            int DeviceCnt = 0;
            char* DeviceName;
            char** CharPath;
            Int32 Option;
            ReEnumerate:
            CharPath = null;
            DeviceName = null;
            Option = 0;
            SelectedDevice = 1;

            DeviceName = (char*)Marshal.AllocCoTaskMem(260 * sizeof(char));

            Console.WriteLine("\n\t0. Exit");
            Console.WriteLine("\t1. Enumerate");

            if (DShowNativeMethods.GetDevicesCount(out DeviceCnt)) // To get the Devices count.
            {
                // Allocating memory to store the instance ID for FSCAM devices.
                CharPath = (char**)Marshal.AllocCoTaskMem(DeviceCnt);
                for (int i = 0; i < DeviceCnt; i++)
                {
                    *(CharPath + i) = (char*)Marshal.AllocCoTaskMem(260 * sizeof(char));
                }

                if (!DShowNativeMethods.GetDeviceName(DeviceName))
                {
                    Console.WriteLine("GetDeviceName Failed..");
                }

                // To get the FSCAM Devices paths
                if (DShowNativeMethods.GetDevicePaths(CharPath))
                {
                    for (int i = 1; i <= DeviceCnt; i++)
                    {
                        Console.WriteLine("\t" + (i + 1) + ". " + Marshal.PtrToStringAuto((IntPtr)DeviceName));
                    }
                    Console.Write("    Choose Device Index : ");

                    Option = int.Parse(Console.ReadLine());

                    if (Option == 0)
                    {
                        Environment.Exit(0); // To Exit the Application
                    }
                    if (Option == 1)
                    {
                        goto ReEnumerate;
                    }
                    if (Option < 1 || Option > DeviceCnt + 1)
                    {
                        Console.WriteLine("\n\tPlease Choose valid Index... :( \n");
                        goto ReEnumerate;
                    }
                    // Incrementing the device path pointer To choose the selected device...
                    for (int i = 2; i < Option; i++)
                    {
                        SelectedDevice++;
                        CharPath++;
                    }


                    RESTART:
                    if (DShowNativeMethods.InitExtensionUnit(*CharPath))
                    {
                        Console.WriteLine("\nSelected device Index: " + SelectedDevice);

                        Console.WriteLine("\t0. Exit");
                        Console.WriteLine("\t1. Read FirmWare Version");
                        Console.WriteLine("\t2. Camera Unique ID");
                        Console.WriteLine("\t3. Reset Device");
                        Console.WriteLine("\t4. Back");

                        Console.Write("     Choose the Option : ");
                        Option = int.Parse(Console.ReadLine());
                        if (Option < 0 || Option > 4)
                        {
                            Console.WriteLine("Invalid Option... \n");
                            goto RESTART;
                        }

                        switch (Option)
                        {
                            case 0:
                                {
                                    // To Exit the Application
                                    Environment.Exit(0);
                                    break;
                                }
                            case 1:
                                {
                                    // Read FW Version
                                    int[] FWVer = new int[4];
                                    if (DShowNativeMethods.ReadFirmwareVersion(out FWVer[0], out FWVer[1], out FWVer[2], out FWVer[3]))
                                    {
                                        Console.Write("\t\t\t FirmWare Version : ");
                                        Console.WriteLine(FWVer[0] + "." + FWVer[1] + "." + FWVer[2] + "." + FWVer[3]);
                                    }
                                    else
                                        Console.WriteLine("Read FirmWare Viersion Is failed ");
                                    break;
                                }
                            case 2:
                                {
                                    char* CamUniqueID;
                                    CamUniqueID = (char*)Marshal.AllocCoTaskMem(260 * sizeof(char));
                                    if (DShowNativeMethods.GetCameraUniqueID(CamUniqueID))
                                    {
                                        Console.WriteLine("\t\t\t Camera Unique ID : " + Marshal.PtrToStringAuto((IntPtr)CamUniqueID));
                                    }
                                }
                                break;

                            case 3:
                                {
                                    if (!DShowNativeMethods.ResetDevice())
                                    {

                                        Console.WriteLine("ResetDevice Failed");
                                    }
                                    else
                                    {
                                        Thread.Sleep(4000);
                                        Console.WriteLine("ResetDevice Success");
                                    }
                                    goto ReEnumerate;

                                }

                            case 4:
                                goto ReEnumerate;
                            default:
                                {
                                    Console.WriteLine("\t Invalied Option...\n");
                                }
                                break;
                        }
                        if (DShowNativeMethods.DeinitExtensionUnit())
                        {
                            goto RESTART;
                        }
                    }
                    else
                    {
                        Console.WriteLine("Failed to Init Extension Unit for selected Device");
                        Console.Read();
                    }
                }
                else
                {
                    Console.WriteLine("Failed to get the Device paths");
                    Console.Read();
                }
            }
            else
            {
                Console.WriteLine(" No devices are found");
                Option = int.Parse(Console.ReadLine());
                goto ReEnumerate;

            }
        }
    }
}